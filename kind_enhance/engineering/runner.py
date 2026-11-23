"""CLI implementation. Search and test intentionally run in different processes."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil
import torch
from kind_enhance.infrastructure.artifacts import STAGES, contact_sheet, save_image, write_csv
from kind_enhance.evaluation.assessment import aggregate, describe
from kind_enhance.evaluation.competition import METRIC_NAMES
from kind_enhance.data.paired_dataset import PairedLowLightDataset
from kind_enhance.inference.pipeline import ROOT, configuration, enhance, load_model, measure
from kind_enhance.inference.calibration import luma
from kind_enhance.infrastructure.runtime import image_on_device, read_checkpoint, resource_snapshot, sha256, write_json
from kind_enhance.training.engine import train
from .guard import DataGuard
from .objective import make_loss
from .policy import (REFERENCES, REFERENCE_MEAN, REFERENCE_STD, benchmark_report, decision,
                     exam_score, propose, quality_failures, validate_settings)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def output_path(path):
    path = path.resolve()
    allowed = (ROOT / 'experiments/engineering_loop').resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError('Output must be a named child of experiments/engineering_loop/')
    return path


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / 'kind_enhance').rglob('*.py'))}


def protected_hashes():
    cfg,_=configuration()
    files = list(dict.fromkeys([ROOT/'configs/model.json',ROOT/'configs/training.json',ROOT/cfg['model'],ROOT/cfg['reference'],ROOT/'data/split-local.json']))
    files += [p for p in (ROOT / 'kind_enhance').rglob('*.py') if 'engineering' not in p.relative_to(ROOT).parts]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def assert_protected(protocol):
    if any(sha256(ROOT / p) != digest for p, digest in protocol['protected_hashes'].items()):
        raise ValueError('Approved main model/configuration/source changed')
    if source_hashes() != protocol['source_hashes']:
        raise ValueError('Search code changed; do not mix protocols in one run')


def validate_split(manifest):
    ids = []
    files = []
    for name, size in (('train', 620), ('val', 69), ('test', 100)):
        rows = manifest['splits'][name]
        if len(rows) != size:
            raise ValueError(f'Unexpected {name} split size')
        ids.append({row['id'] for row in rows})
        files.append({row[key] for row in rows for key in ('low', 'high')})
    if ids[2] != {f'{i:05d}' for i in range(690, 790)}:
        raise ValueError('Official test IDs changed')
    for i in range(3):
        for j in range(i):
            if ids[i] & ids[j] or files[i] & files[j]:
                raise ValueError('Split overlap')


@contextmanager
def process_lock(out):
    path = out / 'process.lock'
    if path.exists():
        previous = read_json(path)
        if psutil.pid_exists(previous['pid']):
            raise RuntimeError('Another process is using this run')
        path.unlink()
    with path.open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid()}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def copy_checkpoint(source, destination):
    temporary = destination.with_suffix('.pending')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def recover_state_checkpoint(out, state):
    """Recover a crash between state commit and atomic checkpoint promotion."""
    if sha256(out / 'best.pt') == state['best_sha256']:
        return
    pending = out / 'next_best.pt'
    if not pending.exists() or sha256(pending) != state['best_sha256']:
        raise ValueError('Incumbent checkpoint differs from saved state')
    pending.replace(out / 'best.pt')


def prepare_protocol(settings, out):
    config, manifest = configuration()
    validate_split(manifest)
    recipe = deepcopy(read_json(ROOT / 'configs/training.json')['config'])
    recipe.pop('loss_source_sha256', None)
    recipe.pop('metrics_source_sha256', None)
    recipe.update(metric_reward_weight=0.0, metric_component_weights=[1.0] * 4, flat_noise_weight=0.0)
    protocol = {'settings': settings, 'output_config': config, 'base_recipe': recipe,
                'source_hashes': source_hashes(), 'protected_hashes': protected_hashes(),
                'manifest_sha256': config['manifest_sha256'], 'reference_methods': REFERENCES,
                'reference_mean': REFERENCE_MEAN.tolist(), 'reference_std_population': REFERENCE_STD.tolist(),
                'selection': 'Fixed-five-reference equal-weight z on validation, with fidelity guards',
                'final_ranking': 'Also recompute all six methods (five baselines + selected) as in slide 3',
                'test_feedback_allowed': False, 'prior_project_test_evaluations_exist': (ROOT / 'results/test/metrics.json').exists()}
    path = out / 'protocol.json'
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError('Frozen protocol differs; use a new run directory')
    else:
        write_json(path, protocol)
        for relative in protocol['source_hashes']:
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        for name in ('engineering_loop', 'training', 'model'):
            target = out / 'source/configs' / f'{name}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == 'engineering_loop':
                write_json(target, settings)
            else:
                shutil.copyfile(ROOT / 'configs' / f'{name}.json', target)
    return protocol, config, manifest


def search(settings, out):
    validate_settings(settings)
    out.mkdir(parents=True, exist_ok=True)
    with process_lock(out):
        if (out / 'selection.json').exists() or (out / 'test/test_lock.json').exists():
            raise FileExistsError('Selection is sealed; search cannot continue in this run')
        protocol, config, manifest = prepare_protocol(settings, out)
        guard = DataGuard(manifest['source']['data_root'], manifest, ('train', 'val'))
        guard.install()
        teacher, _, _ = load_model()
        teacher.requires_grad_(False).eval()
        training = PairedLowLightDataset(manifest['source']['data_root'], manifest['splits']['train'],
                                        patch=protocol['base_recipe']['patch'], cache=True)
        validation = PairedLowLightDataset(manifest['source']['data_root'], manifest['splits']['val'], cache=True)
        state_path = out / 'state.json'
        started = time.perf_counter()
        state = None
        status = 'failed'
        try:
            if state_path.exists():
                state = read_json(state_path)
                recover_state_checkpoint(out, state)
                for trial in state['trials']:
                    for filename in ('best.pt', 'last.pt'):
                        (out / 'trials' / f"{trial['id']:03d}" / filename).unlink(missing_ok=True)
            else:
                baseline, rows = measure(teacher, validation, config, collect=True)
                write_csv(out / 'baseline_validation.csv', rows)
                copy_checkpoint(ROOT / config['model'], out / 'best.pt')
                state = {'baseline': baseline, 'best_metrics': baseline, 'best_trial': 'baseline',
                         'best_sha256': sha256(out / 'best.pt'), 'trials': [], 'pending': None,
                         'elapsed_seconds': 0.0, 'consecutive_rejections': 0}
                write_json(state_path, state)
                print(json.dumps({'baseline_validation': baseline}), flush=True)
            previous_seconds = state['elapsed_seconds']
            stop_reason = 'trial_budget'
            while len(state['trials']) < settings['max_trials']:
                elapsed = previous_seconds + time.perf_counter() - started
                if not state['pending'] and elapsed >= settings['max_minutes'] * 60:
                    stop_reason = 'time_budget'
                    break
                if not state['pending'] and state['consecutive_rejections'] >= settings['patience']:
                    stop_reason = 'validation_patience'
                    break
                if state['pending'] is None:
                    recipe = propose(protocol['base_recipe'], state['trials'], settings)
                    recipe['initial_sha256'] = state['best_sha256']
                    recipe['loss_recipe'] = 'engineering_fixed_z_v1'
                    state['pending'] = {'id': len(state['trials']) + 1, 'recipe': recipe,
                                        'parent_sha256': state['best_sha256'], 'parent_metrics': state['best_metrics']}
                    write_json(state_path, state)
                pending = state['pending']
                trial_dir = out / 'trials' / f"{pending['id']:03d}"
                recipe = pending['recipe']
                torch.manual_seed(recipe['seed'])
                model, _, _ = load_model()
                initial = read_checkpoint(out / 'best.pt', 'cpu', config['manifest_sha256'], 'enhance')
                model.load_state_dict(initial['model'], strict=True)
                def validate(current, dataset, device):
                    values, rows = measure(current, dataset, config, collect=True)
                    values['quality_failures'] = quality_failures(values, state['baseline'], settings['quality_limits'])
                    write_csv(trial_dir / 'validation.csv', rows)
                    return values, (not values['quality_failures'], exam_score(values), values['PSNR'])
                print(json.dumps({'trial_start': pending['id'], 'recipe': recipe}), flush=True)
                result = train(model, training, validation, torch.device('cpu'), 'enhance', recipe,
                               settings['epochs_per_trial'], trial_dir, config['manifest_sha256'],
                               validation_scoring=validate, loss_function=make_loss(teacher, config))
                metrics = result['best']['validation']
                verdict = decision(metrics, state['best_metrics'], state['baseline'], settings)
                record = {**pending, 'metrics': metrics, 'decision': verdict, 'best_epoch': result['best']['epoch']}
                # Commit next state before retirement. Pending copy enables consistent recovery.
                if verdict['kept']:
                    copy_checkpoint(Path(result['checkpoint']), out / 'next_best.pt')
                    state['best_sha256'] = sha256(out / 'next_best.pt')
                    state['best_metrics'], state['best_trial'] = metrics, pending['id']
                    state['consecutive_rejections'] = 0
                else:
                    state['consecutive_rejections'] += 1
                result['decision'] = verdict
                result['checkpoint_retired_after_commit'] = True
                write_json(trial_dir / 'result.json', result)
                state['trials'].append(record)
                state['pending'] = None
                state['elapsed_seconds'] = previous_seconds + time.perf_counter() - started
                write_json(state_path, state)
                if verdict['kept']:
                    (out / 'next_best.pt').replace(out / 'best.pt')
                for filename in ('best.pt', 'last.pt'):
                    target = (trial_dir / filename).resolve()
                    if not target.is_relative_to(out.resolve()):
                        raise ValueError('Checkpoint retirement escaped the run directory')
                    target.unlink(missing_ok=True)
                print(json.dumps({'trial_end': pending['id'], 'decision': verdict,
                                  'incumbent_validation_z': exam_score(state['best_metrics'])}), flush=True)
                assert_protected(protocol)
            # Reproduce the selected validation measurements before exposing test to another process.
            selected, _, _ = load_model()
            selected.load_state_dict(read_checkpoint(out / 'best.pt', 'cpu', config['manifest_sha256'], 'enhance')['model'], strict=True)
            reproduced, rows = measure(selected, validation, config, collect=True)
            if any(abs(reproduced[k] - state['best_metrics'][k]) > 1e-7 for k in (*METRIC_NAMES, 'PSNR', 'SSIM')):
                raise ValueError('Selected checkpoint failed validation reproduction')
            write_csv(out / 'selected_validation.csv', rows)
            assert_protected(protocol)
            selection = {'selected_trial': state['best_trial'], 'checkpoint': 'best.pt',
                         'checkpoint_sha256': sha256(out / 'best.pt'), 'validation': reproduced,
                         'validation_benchmark_reference_only': benchmark_report(reproduced),
                         'calibration': config['calibration'], 'manifest_sha256': config['manifest_sha256'],
                         'protocol_sha256': sha256(out / 'protocol.json'), 'stop_reason': stop_reason,
                         'selected_on': 'validation_only', 'test_read_during_search': False,
                         'main_model_promoted': False, 'search_sealed': True}
            write_json(out / 'selection.json', selection)
            write_search_report(out, state, selection)
            status = 'completed'
            print(json.dumps({'search_completed': selection}), flush=True)
        finally:
            seconds = time.perf_counter() - started
            write_json(out / 'search_data_access.json', {**guard.report(), 'status': status,
                'session_seconds': seconds, 'resources': resource_snapshot(torch.device('cpu')),
                'main_unchanged': protected_hashes() == protocol['protected_hashes']})


def write_search_report(out, state, selection):
    lines = []
    for name, metrics, verdict in [('baseline', state['baseline'], 'giữ ban đầu'),
                                  *[(str(t['id']), t['metrics'], 'giữ' if t['decision']['kept'] else 'loại: ' + t['decision']['reason'])
                                    for t in state['trials']]]:
        values = ' | '.join(f'{metrics[k]:.6f}' for k in METRIC_NAMES)
        lines.append(f'| {name} | {values} | {exam_score(metrics):.6f} | {verdict} |')
    (out / 'VALIDATION_REPORT.md').write_text(f'''# Engineering loop: validation

620 cặp train, 69 validation. Không mở ảnh test. Chọn bằng Z tham chiếu cố định 5 phương pháp trong đề, trọng số 1/4 cho từng MLI/CI/E/AG; điều kiện chất lượng khóa trong protocol.json. PSNR/SSIM, màu và clipping chỉ làm ràng buộc. Không chặn MLI/E ở dải cũ để cả bốn chỉ số có thể thay đổi. Proxy entropy32 chỉ dùng gradient huấn luyện; số báo cáo luôn tính entropy256 từ ảnh uint8.

| Lượt | MLI | CI | E | AG | Z tham chiếu | Quyết định |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(lines)}

Lượt được chọn: **{selection['selected_trial']}**. Dừng: **{selection['stop_reason']}**. Giữ best.pt; checkpoint các lượt đã hoàn thành được loại sau khi ghi log. Checkpoint được duyệt trong configs/model.json không thay đổi. Đây là tìm kiếm cục bộ có giới hạn, không phải tự sửa mã hoặc kiến trúc và không bảo đảm tăng sau mỗi lượt.

So sánh AWF ở validation chỉ là tham chiếu bảng đề trên tập khác. Báo cáo test được thực hiện bởi tiến trình riêng sau selection.json, có cả Z cố định và bảng chuẩn hóa lại 6 phương pháp; không lấy hai cách chuẩn hóa so chéo. Tập test dự án đã từng được đánh giá/tinh chỉnh lựa chọn ở các lượt lịch sử, nên không gọi kết quả mới là phép kiểm tra hoàn toàn độc lập.
''', encoding='utf-8')


def finalize(out):
    protocol = read_json(out / 'protocol.json')
    assert_protected(protocol)
    selection = read_json(out / 'selection.json')
    state = read_json(out / 'state.json')
    search_access = read_json(out / 'search_data_access.json')
    if search_access['status'] != 'completed' or search_access['test_images_allowed'] or search_access['blocked_attempts']:
        raise ValueError('A clean, completed search audit is required')
    if not selection['search_sealed'] or selection['selected_on'] != 'validation_only' or state['pending']:
        raise ValueError('Search must be complete before test evaluation')
    if sha256(out / 'protocol.json') != selection['protocol_sha256'] or sha256(out / 'best.pt') != selection['checkpoint_sha256']:
        raise ValueError('Locked configuration or selected checkpoint changed')
    test_dir = out / 'test'
    test_dir.mkdir(exist_ok=True)
    with process_lock(out):
        if (test_dir / 'metrics.json').exists():
            print('Test already completed; no further search or test evaluation.', flush=True)
            return
        lock = {'checkpoint_sha256': selection['checkpoint_sha256'], 'selection_sha256': sha256(out / 'selection.json'),
                'protocol_sha256': selection['protocol_sha256'], 'calibration': selection['calibration'],
                'test_count': 100, 'feedback_to_search': False}
        lock_path = test_dir / 'test_lock.json'
        if lock_path.exists():
            if read_json(lock_path) != lock:
                raise ValueError('Cannot change a partially completed test evaluation')
        else:
            write_json(lock_path, lock)
        config, manifest = configuration()
        guard = DataGuard(manifest['source']['data_root'], manifest, ('test',))
        guard.install()
        selected, _, _ = load_model()
        selected.load_state_dict(read_checkpoint(out / 'best.pt', 'cpu', config['manifest_sha256'], 'enhance')['model'], strict=True)
        baseline, _, _ = load_model()
        selected.eval()
        baseline.eval()
        dataset = PairedLowLightDataset(manifest['source']['data_root'], manifest['splits']['test'])
        records = {'selected': [], 'approved_baseline': []}
        image_dir = test_dir / 'images'
        image_dir.mkdir(exist_ok=True)
        started = time.perf_counter()
        status = 'failed'
        try:
            with torch.inference_mode():
                for index, (low, high, image_id) in enumerate(dataset):
                    low, high = [image_on_device(x.unsqueeze(0), 'cpu') for x in (low, high)]
                    pred = enhance(selected, low, config)
                    original = pred if selection['checkpoint_sha256'] == config['model_sha256'] else enhance(baseline, low, config)
                    for name, output in (('selected', pred), ('approved_baseline', original)):
                        records[name].append({'id': image_id, **describe(output, high),
                            'chroma_mae': torch.nn.functional.l1_loss(output-luma(output), high-luma(high)).item()})
                    save_image(image_dir / f'{image_id}.png', pred)
                    if index in (0, 49, 99):
                        stages = test_dir / 'stages' / image_id
                        stages.mkdir(parents=True, exist_ok=True)
                        r, light = selected.decomposition(low)
                        raw, restored, adjusted, _ = selected.enhance(r, light)
                        for (name, _), value in zip(STAGES, (low, r, light, r*light, restored, adjusted, raw, high)):
                            save_image(stages / f'{name}.png', value)
                        save_image(stages / '09_final_calibrated.png', pred)
                        contact_sheet(stages, 'KinD + Attention U-Net + Curve CNN | ' + image_id)
                        save_image(test_dir / f'{image_id}_comparison.png', torch.cat((low, original, pred, high), dim=-1))
                    if (index+1) % 25 == 0:
                        print(f'Locked test: {index+1}/100', flush=True)
            results = {}
            for name, rows in records.items():
                mean = aggregate(rows)
                mean['chroma_mae'] = statistics.mean(row['chroma_mae'] for row in rows)
                results[name] = {'mean': mean, 'sample_std': {key: statistics.stdev(row[key] for row in rows)
                    for key in (*METRIC_NAMES, 'PSNR', 'SSIM')}, 'benchmark': benchmark_report(mean)}
                write_csv(test_dir / f'{name}_per_image.csv', rows)
            assert_protected(protocol)
            if sha256(out / 'selection.json') != lock['selection_sha256']:
                raise ValueError('Selection changed during test')
            report = {'methods': results, 'seconds': time.perf_counter()-started,
                      'resources': resource_snapshot(torch.device('cpu')), 'main_model_promoted': False,
                      'test_used_for_training_or_search': False, 'prior_project_test_evaluations_exist':
                      protocol['prior_project_test_evaluations_exist'], 'selection_checkpoint_sha256': selection['checkpoint_sha256']}
            write_test_report(test_dir, report)
            write_json(test_dir / 'metrics.json', report)
            status = 'completed'
            print(json.dumps({'test_completed': results}), flush=True)
        finally:
            write_json(test_dir / 'data_access.json', {**guard.report(), 'status': status,
                       'main_unchanged': protected_hashes() == protocol['protected_hashes']})


def write_test_report(out, results):
    selected = results['methods']['selected']
    lines = []
    for name, item in results['methods'].items():
        cells = ' | '.join(f"{item['mean'][key]:.6f}" for key in (*METRIC_NAMES, 'PSNR', 'SSIM'))
        lines.append(f"| {name} | {cells} | {item['benchmark']['fixed_reference_z']:.6f} |")
    awf = REFERENCES['AWF']
    lines.append('| AWF (bảng đề) | ' + ' | '.join(f'{v:.6f}' for v in awf) +
                 f" | — | — | {selected['benchmark']['fixed_reference_AWF_z']:.6f} |")
    cohort = selected['benchmark']['six_method_cohort']
    ranking = '\n'.join(f"| {name} | {cohort['composite_z'][name]:.6f} | {cohort['rank'][name]} |"
                        for name in sorted(cohort['methods'], key=lambda n: cohort['rank'][n]))
    margin = selected['benchmark']['six_method_margin_vs_AWF']
    conclusion = 'cao hơn' if margin > 0 else ('thấp hơn' if margin < 0 else 'bằng')
    (out / 'BAO_CAO_TEST_100.md').write_text(f'''# Kết quả thực đo trên 100 ảnh test

Checkpoint đã khóa bằng validation trước khi tiến trình này mở test. Không train, không backward, không chọn lại checkpoint và không đưa kết quả test vào bộ đề xuất cấu hình. Bản chính giữ nguyên.

| Phương pháp | MLI | CI | E | AG | PSNR | SSIM | Z tham chiếu cố định |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Z tham chiếu dùng μ/σ từ 5 phương pháp trong đề để số giữa các lượt có cùng thước đo. Điểm AWF tính lại từ số làm tròn trong đề, gần +1.4479; mốc slide 5 là >1.45. Không lấy Z này so chéo với Z của nhóm 6 bên dưới.

## Chuẩn hóa lại toàn bộ nhóm 6 phương pháp theo slide 3

Nhóm gồm 5 phương pháp trong đề + Candidate, không tính Input; trung bình bốn z-score, độ lệch chuẩn giữa các phương pháp dùng ddof=0.

| Phương pháp | Z trong cùng nhóm 6 | Hạng |
|---|---:|---:|
{ranking}

Trong nhóm này Candidate **{conclusion} AWF**, chênh **{margin:+.6f}**. Đây là so sánh với số AWF công bố trong đề: chưa có mã chấm/AWF gốc để xác nhận công thức đo và tập ảnh khớp tuyệt đối, nên chưa gọi là chứng nhận chính thức.

Thời gian test {results['seconds']:.2f} giây; tài nguyên thực đo trong metrics.json. 100 ảnh đầu ra ở images/, điểm từng ảnh và SD mẫu trong CSV/JSON. Ba ví dụ ở stages/ có Input, R, L, R×L, R′, L′, đầu ra mạng, GT và ảnh hiệu chỉnh cuối; 00_pipeline.png trình bày 8 thành phần đầu. Ảnh *_comparison.png: Input / bản chốt / bản loop / GT.

Giới hạn: dự án đã có các lần đánh giá và lựa chọn theo test trong lịch sử ({results['prior_project_test_evaluations_exist']}); test này không phải tập mới hoàn toàn độc lập. Lượt loop hiện tại tách train/validation/test bằng allowlist và tiến trình. Tìm kiếm ngắn trên CPU, một seed; không bảo đảm khái quát hoặc cải thiện ở mọi chỉ số. Kết quả test là điểm kết thúc của lượt này, kể cả khi thua AWF.
''', encoding='utf-8')


def smoke(settings, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol, config, manifest = prepare_protocol(settings, out)
    guard = DataGuard(manifest['source']['data_root'], manifest, ('train', 'val'))
    guard.install()
    model, _, _ = load_model()
    teacher, _, _ = load_model()
    teacher.requires_grad_(False).eval()
    recipe = propose(protocol['base_recipe'], [], settings)
    recipe.update(metric_reward_weight=settings['initial_metric_reward'], flat_noise_weight=1.0)
    dataset = PairedLowLightDataset(manifest['source']['data_root'], manifest['splits']['train'][:2], patch=80)
    batch = [dataset[i] for i in range(2)]
    low, high = [image_on_device(torch.stack([row[i] for row in batch]), 'cpu') for i in (0, 1)]
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['lr'])
    model.train()
    before = next(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = make_loss(teacher, config)(model, low, high, recipe)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    updated = not torch.equal(before, next(model.parameters()).detach())
    validation = PairedLowLightDataset(manifest['source']['data_root'], manifest['splits']['val'][:2])
    measured = measure(model, validation, config)
    assert updated and torch.isfinite(norm)
    assert_protected(protocol)
    write_json(out / 'smoke.json', {'passed': True, 'loss': loss.item(), 'gradient_norm_before_clip': norm.item(),
        'model_weights_updated': updated, 'validation_subset_only': measured, 'data_access': guard.report(),
        'main_unchanged': True, 'not_a_benchmark_result': True})
    print('Real-data smoke test passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'search', 'finalize', 'smoke'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/engineering_loop.json')
    args = parser.parse_args()
    out = output_path(args.output)
    settings = read_json(args.config)
    validate_settings(settings)
    if args.command in ('run', 'finalize') and (out / 'selection.json').exists():
        if settings != read_json(out / 'protocol.json')['settings']:
            raise ValueError('Final evaluation must use the frozen settings')
    torch.set_num_threads(settings['threads'])
    torch.set_num_interop_threads(1)
    if args.command == 'run':
        # Search's audit hook can never be disabled to read test; use distinct child processes.
        if not (out / 'selection.json').exists():
            subprocess.run([sys.executable, '-B', '-u', '-m', 'kind_enhance.engineering', 'search',
                            '--output', str(out), '--config', str(args.config.resolve())], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-B', '-u', '-m', 'kind_enhance.engineering', 'finalize',
                        '--output', str(out), '--config', str(args.config.resolve())], cwd=ROOT, check=True)
    elif args.command == 'search':
        search(settings, out)
    elif args.command == 'finalize':
        finalize(out)
    else:
        smoke(settings, out)


if __name__ == '__main__':
    main()

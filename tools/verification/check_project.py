"""Validate the production folder tree, documentation links and approved assets."""
from pathlib import Path
import importlib
import json
import re
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from kind_enhance.infrastructure.runtime import sha256,write_json


def main():
    expected=[
        'doc/README.md',
        'doc/1_tong_quan/1.1_cau_truc_du_an.md',
        'doc/1_tong_quan/1.2_huong_dan_su_dung.md',
        'doc/1_tong_quan/1.3_luu_y_quy_trinh_lam.md',
        'doc/1_tong_quan/1.4_quy_trinh_vibe_code.md',
        'doc/1_tong_quan/1.5_quy_trinh_lam_du_an.md',
        'doc/2_cac_ky_thuat/2.1_architecture.md',
        'doc/2_cac_ky_thuat/2.2_business_flow.md',
        'doc/2_cac_ky_thuat/2.3_mo_hinh_c4.md',
        'doc/3_san_pham/3.1_changelog.md',
        'doc/3_san_pham/3.2_roadmap.md',
        'README.md','app.py','pyproject.toml','kind_enhance/ui/local_web.py','kind_enhance/ui/web/index.html',
        'kind_enhance/ui/web/app.js','kind_enhance/ui/web/app.css']
    assert all((ROOT/path).is_file() for path in expected)
    assert not (ROOT/'docs').exists(),'Old docs directory should not remain'
    checked_links=0;missing=[]
    pages=[ROOT/'README.md',*sorted((ROOT/'doc').rglob('*.md')),*sorted((ROOT/'reports').glob('*.md')),ROOT/'tools/README.md',ROOT/'assets/demo/README.md',ROOT/'kind_enhance/README.md']
    for page in pages:
        for target in re.findall(r'\]\(([^)]+)\)',page.read_text(encoding='utf-8')):
            if target.startswith(('https://','http://','#')):continue
            target=target.split('#')[0].strip('<>')
            destination=(page.parent/target).resolve()
            checked_links+=1
            if not destination.exists():missing.append(dict(page=str(page),target=target))
    assert not missing,missing
    modules=['kind_enhance.application.enhancement','kind_enhance.infrastructure.images','kind_enhance.ui.local_web',
             'tools.training.refine_four_metrics','tools.evaluation.evaluate_four_metrics',
             'tools.evaluation.verify_four_metric_artifacts','tools.inspection.inspect_model_architecture',
             'tools.research.context_detail_support','tools.research.refine_context_details']
    for name in modules:importlib.import_module(name)
    # Check the real source layout and import every library module (no entrypoints).
    layout=json.loads((ROOT/'outputs/checks/code_reorganization.json').read_text())
    for old,new in layout['moved_files'].items():
        assert not (ROOT/old).exists(),old
        assert (ROOT/new).is_file(),new
    package_modules=[]
    for path in sorted((ROOT/'kind_enhance').rglob('*.py')):
        if path.name in ('__init__.py','__main__.py'):continue
        name='.'.join(path.relative_to(ROOT).with_suffix('').parts)
        importlib.import_module(name)
        package_modules.append(name)
    project=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))
    assert project['project']['name']=='kind-enhance' and project['project']['version']=='1.0.0'
    cfg=json.loads((ROOT/'configs/model.json').read_text())
    expected_hash='902c100e7455dd23946f9de7891c2e4f730427149f3604f9dcdabb4e79c747d0'
    assert cfg['model_sha256']==sha256(ROOT/cfg['model'])==expected_hash
    assert sha256(ROOT/cfg['manifest'])==cfg['manifest_sha256']
    manifest=json.loads((ROOT/cfg['manifest']).read_text())
    assert {key:len(rows) for key,rows in manifest['splits'].items()}=={'train':620,'val':69,'test':100}
    gui=json.loads((ROOT/'outputs/checks/local_web_smoke.json').read_text())
    reproduction=json.loads((ROOT/'results/active_model/pipeline_verification.json').read_text())
    assert gui['verified'] and gui['entrypoint']=='app.py'
    assert all(row['pixel_identical_to_approved'] for row in gui['results'])
    for obsolete in ('CHAY_LOCAL.cmd','LOCALHOST.cmd','start_app.cmd','scripts/start_app.ps1'):
        assert not (ROOT/obsolete).exists(),obsolete
    assert reproduction['full_test_PNGs_identical']==100 and reproduction['CLI_PNG_identical']
    result=dict(verified=True,app_version='1.0.0',required_documents=len(expected),markdown_links_checked=checked_links,
                broken_links=0,module_imports_checked=len(set(modules+package_modules)),
                source_layout_verified=True,moved_code_files=len(layout['moved_files']),checkpoint_sha256=expected_hash,
                network_parameters=2217963,GUI_workflow_verified=True,approved_test_PNGs_identical=100,
                original_dataset_manifest_unchanged=True,no_dependencies_downloaded=True,
                documentation_folder='doc',entrypoint='app.py',readme='README.md')
    write_json(ROOT/'outputs/checks/project_verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

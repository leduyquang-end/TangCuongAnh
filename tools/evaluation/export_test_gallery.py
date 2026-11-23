"""Publish a browsable copy of the approved 100 test exports; never train or infer."""
from pathlib import Path
import csv
import html
import json
import os
import shutil
import sys
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from PIL import Image
from kind_enhance.inference.pipeline import configuration
from kind_enhance.infrastructure.runtime import sha256,write_json


def main():
    source=ROOT/'experiments/ag_refinement/run_four_metric_curriculum/test'
    destination=ROOT/'outputs/test_100'
    images=destination/'images'
    cfg,manifest=configuration()
    metrics=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    assert sha256(ROOT/cfg['model'])==cfg['model_sha256']==metrics['checkpoint_sha256']
    assert sha256(ROOT/cfg['manifest'])==cfg['manifest_sha256']
    pairs=manifest['splits']['test']
    assert len(pairs)==100 and len({p['id'] for p in pairs})==100
    with (source/'per_image.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows={row['id']:row for row in csv.DictReader(handle)}
    assert set(rows)=={p['id'] for p in pairs}
    expected={p['id']+'_RGB.png' for p in pairs}
    assert {p.name for p in (source/'images').glob('*.png')}==expected
    images.mkdir(parents=True,exist_ok=True)
    assert not {p.name for p in images.glob('*.png')}-expected,'Unexpected images in output directory'
    evidence=[];cards=[]
    def local_url(path):
        return html.escape(quote(os.path.relpath(path,destination).replace(os.sep,'/'),safe='/'),quote=True)
    for index,pair in enumerate(pairs,1):
        name=pair['id']+'_RGB.png'
        original=source/'images'/name
        exported=images/name
        original_hash=sha256(original)
        if exported.exists():
            assert sha256(exported)==original_hash,f'Existing output differs: {exported}'
        else:
            shutil.copy2(original,exported)
        assert sha256(exported)==original_hash
        with Image.open(exported) as picture:
            picture.verify()
        with Image.open(exported) as picture:
            assert picture.mode=='RGB' and picture.size==(600,400)
        figures=[]
        low=Path(manifest['source']['data_root'])/pair['low']
        high=Path(manifest['source']['data_root'])/pair['high']
        for label,path in [('Ảnh tối đầu vào',low),('Kết quả mô hình',exported),('Ảnh chuẩn (GT)',high)]:
            assert path.is_file(),path
            url=local_url(path)
            figures.append(f'<figure><figcaption>{label}</figcaption><a href="{url}" target="_blank" rel="noopener"><img loading="lazy" src="{url}" alt="{label} {pair["id"]}" width="600" height="400"></a></figure>')
        row=rows[pair['id']]
        values=' · '.join(f'{key}: {float(row[key]):.4f}' for key in ('MLI','CI','E','AG','PSNR','SSIM'))
        cards.append(f'<article data-id="{pair["id"]}"><h2>{index:03d}/100 — {pair["id"]}</h2><div class="triplet">{"".join(figures)}</div><p class="measure">{values}</p></article>')
        evidence.append(dict(id=pair['id'],file='images/'+name,sha256=original_hash,mode='RGB',size=[600,400]))
    for name in ('per_image.csv','metrics.json','comparison.png'):
        shutil.copy2(source/name,destination/name)
    means=metrics['candidate']
    summary=' · '.join(f'{key}: {means[key]:.6f}' for key in ('MLI','CI','E','AG','PSNR','SSIM'))
    report=local_url(ROOT/cfg['evaluation_report'])
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100 ảnh test — KinD</title>
<style>body{margin:0;background:#edf1f6;color:#17243a;font:16px/1.55 system-ui,sans-serif}header,main{max-width:1380px;margin:auto;padding:24px}header{padding-bottom:0}h1{margin:0}a{color:#145caa}article{background:white;border:1px solid #d9e1ed;border-radius:12px;padding:18px;margin-bottom:22px}h2{font-size:20px;margin:0 0 14px}.triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}figcaption{font-weight:600;margin-bottom:8px}img{display:block;width:100%;height:auto;aspect-ratio:3/2;object-fit:contain;background:#101722;border-radius:6px}.measure{font-size:14px;color:#465674;margin-bottom:0}.note{color:#465674}input{padding:10px 12px;border:1px solid #a7b5c8;border-radius:6px;font:inherit;width:230px}@media(max-width:720px){.triplet{grid-template-columns:1fr}header,main{padding:16px}}</style>
<header><h1>100 ảnh test của bản đã chốt</h1><p>Đủ 100 ảnh PNG RGB, ID 00690–00789. Nhấn ảnh để xem kích thước đầy đủ.</p><p>SUMMARY</p><p><a href="images/">Thư mục 100 ảnh đầu ra</a> · <a href="per_image.csv">Chỉ số từng ảnh (CSV)</a> · <a href="REPORT">Báo cáo so sánh AWF</a></p><p class="note">Đây là bản sao có kiểm tra hash từ kết quả đã xuất. Không huấn luyện, không tinh chỉnh và không đánh giá lại để chọn tham số. GT chỉ hiển thị để đối chiếu.</p><label for="filter">Tìm ID ảnh: </label><input id="filter" type="search" placeholder="Ví dụ: 00690"><span id="count"> 100/100 ảnh</span></header><main>CARDS</main>
<script>const input=document.getElementById('filter'),cards=[...document.querySelectorAll('article')];input.addEventListener('input',()=>{let shown=0;for(const card of cards){card.hidden=!card.dataset.id.includes(input.value.trim());if(!card.hidden)shown++;}document.getElementById('count').textContent=' '+shown+'/100 ảnh';});</script></html>'''
    page=page.replace('SUMMARY',summary).replace('REPORT',report).replace('CARDS','\n'.join(cards))
    (destination/'index.html').write_text(page,encoding='utf-8')
    readme='''# Bộ 100 ảnh test đã chốt

- Mở [index.html](index.html) để xem đủ 100 bộ ảnh tối → kết quả → GT, có tìm theo ID.
- Thư mục [images](images) có đúng 100 ảnh đầu ra PNG RGB, từ 00690_RGB.png đến 00789_RGB.png.
- [per_image.csv](per_image.csv): chỉ số từng ảnh; [metrics.json](metrics.json): số liệu gốc.
- [export_manifest.json](export_manifest.json): nguồn và SHA256 của từng ảnh sao chép.

Bộ này được sao chép từ experiments/ag_refinement/run_four_metric_curriculum/test. Không huấn luyện lại, không tinh chỉnh từ test. Ảnh tối/GT trong trang xem được tham chiếu tới dataset cục bộ; cần giữ dataset ở vị trí hiện tại. Bản PNG kết quả trong images có thể mang đi độc lập.
'''
    (destination/'README.md').write_text(readme,encoding='utf-8')
    assert {p.name for p in images.glob('*.png')}==expected
    output=dict(verified=True,images=100,first_id=pairs[0]['id'],last_id=pairs[-1]['id'],checkpoint_sha256=cfg['model_sha256'],source=source.relative_to(ROOT).as_posix(),destination=destination.relative_to(ROOT).as_posix(),copied_existing_approved_exports=True,all_images_byte_identical_to_source=True,new_inference_performed=False,training_performed=False,parameter_selection_performed=False,mean=means,files=evidence)
    write_json(destination/'export_manifest.json',output)
    print(json.dumps({key:value for key,value in output.items() if key not in ('files','mean')},ensure_ascii=False))


if __name__=='__main__':main()

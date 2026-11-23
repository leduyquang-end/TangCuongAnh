"""Trace the deployed architecture using synthetic inputs; never train or read dataset images."""
import json
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.infrastructure.runtime import sha256

def main():
    torch.set_num_threads(6)
    model,config,_=load_model()
    protected=[ROOT/config['model'],ROOT/'configs/model.json',*list((ROOT/'kind_enhance').rglob('*.py'))]
    hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in protected}
    traces={}
    def shapes(value):
        if isinstance(value,torch.Tensor):return list(value.shape)
        return [shapes(v) for v in value]
    for name,shape in [('image_400x600',(1,3,400,600)),('train_patch_80x80',(8,3,80,80)),('odd_81x79',(1,3,81,79))]:
        rows=[];handles=[]
        def register(path,layer):
            def hook(module,inputs,output):
                rows.append({'name':path,'type':type(module).__name__,'input':shapes(inputs),
                    'output':shapes(output),'kernel':list(module.kernel_size),'stride':list(module.stride),
                    'padding':list(module.padding),'dilation':list(module.dilation),'groups':module.groups,
                    'bias':module.bias is not None,'parameters':sum(p.numel() for p in module.parameters(recurse=False))})
            handles.append(layer.register_forward_hook(hook))
        for path,layer in model.named_modules():
            if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):register(path,layer)
        torch.manual_seed(827)
        x=torch.rand(shape).to(memory_format=torch.channels_last)
        with torch.inference_mode():y=model(x)
        for handle in handles:handle.remove()
        with torch.inference_mode():final=enhance(model,x,config)
        assert tuple(y.shape)==shape and tuple(final.shape)==shape
        assert torch.isfinite(y).all() and torch.isfinite(final).all()
        assert sum(r['parameters'] for r in rows)==sum(p.numel() for p in model.parameters())
        traces[name]={'input':list(shape),'raw_output':list(y.shape),'final_output':list(final.shape),'layers':rows}
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}
    result={'checkpoint_sha256':config['model_sha256'],'config':config,'counts':counts,'total':sum(counts.values()),
            'traces':traces,'source_hashes':hashes,'synthetic_inputs_only':True,'training_performed':False}
    assert all(sha256(ROOT/p)==value for p,value in hashes.items())
    out=ROOT/'doc/2_cac_ky_thuat/phu_luc';out.mkdir(exist_ok=True)
    (out/'model_trace.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Phụ lục: toàn bộ tầng có tham số — đo trực tiếp bằng forward hook','',
           'Sinh bởi `python -B tools/inspection/inspect_model_architecture.py`. Input tổng hợp, không đọc train/validation/test, không cập nhật trọng số.',
           '',f"Checkpoint SHA-256: `{config['model_sha256']}`.",'',
           'Thứ tự bảng là thứ tự thực thi forward. Kích thước tại Conv là trước activation; ConvTranspose là trước thao tác cắt. Pooling, concat, activation dạng functional, nhân/clip không có tham số nên được mô tả trong tài liệu kiến trúc chính.', '',
           'Quy ước shape N×C×H×W; k/s/p là kernel/stride/padding; mọi convolution có dilation=1, groups=1.', '']
    def fmt(shape):return '×'.join(map(str,shape))
    full=traces['image_400x600']['layers'];patch=traces['train_patch_80x80']['layers'];odd=traces['odd_81x79']['layers']
    lines += ['| # | Module đúng tên mã | Loại | k/s/p | Bias | Input ảnh đầy đủ | Output ảnh đầy đủ | Output batch crop | Output kích thước lẻ | Tham số |',
              '|---:|---|---|---|---|---|---|---|---|---:|']
    for i,(a,b,c) in enumerate(zip(full,patch,odd),1):
        assert a['name']==b['name']==c['name']
        spec='/'.join(fmt(a[k]) for k in ('kernel','stride','padding'))
        lines.append(f"| {i} | `{a['name']}` | {a['type']} | {spec} | {a['bias']} | {fmt(a['input'][0])} | {fmt(a['output'])} | {fmt(b['output'])} | {fmt(c['output'])} | {a['parameters']:,} |")
    lines += ['',f"Tổng: **{len(full)} tầng convolution/transposed convolution**, **{sum(counts.values()):,} tham số**.",'',
              '[Tài liệu kiến trúc chính](../2.1_architecture.md) · [Dữ liệu trace đầy đủ](model_trace.json)']
    (ROOT/'doc/2_cac_ky_thuat/phu_luc/52_tang.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':counts,'total':sum(counts.values()),'convolution_layers':len(full),'shapes_checked':list(traces),'protected_files_unchanged':True}))

if __name__=='__main__':main()

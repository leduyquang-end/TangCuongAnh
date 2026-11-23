"""User entrypoints for desktop, photo enhancement, evaluation and training."""
import argparse
import json
from pathlib import Path
import sys
import torch
from kind_enhance.inference.pipeline import ROOT,infer,evaluate,load_inference_model


def build_parser():
    parser=argparse.ArgumentParser(description='KinD — ứng dụng tăng sáng ảnh RGB')
    parser.add_argument('--threads',type=int,default=6,help='Số luồng CPU, mặc định 6')
    commands=parser.add_subparsers(dest='command',required=True)
    gui=commands.add_parser('gui',help='Mở giao diện chọn ảnh và tăng sáng')
    gui.add_argument('--input',type=Path,help='Ảnh được chọn sẵn khi mở giao diện')
    web=commands.add_parser('web',help='Mở giao diện web trên localhost')
    web.add_argument('--port',type=int,default=7860)
    web.add_argument('--no-browser',action='store_true',help='Không tự mở trình duyệt')
    photo=commands.add_parser('infer',help='Tăng sáng một ảnh bằng checkpoint đã chốt')
    photo.add_argument('--input',type=Path,required=True)
    photo.add_argument('--output',type=Path,help='PNG đầu ra; bỏ trống để tự lưu vào outputs/images')
    photo.add_argument('--stages',type=Path,help='Thư mục mới để lưu R/L và các bước xử lý')
    photo.add_argument('--max-side',type=int,default=1280,help='Cạnh dài tối đa; 0 giữ kích thước gốc')
    photo.add_argument('--overwrite',action='store_true',help='Cho phép ghi đè tệp kết quả, luôn bảo vệ ảnh gốc')
    check=commands.add_parser('doctor',help='Kiểm tra Python, checkpoint và cấu hình suy luận')
    evaluate_command=commands.add_parser('evaluate',help='Đánh giá checkpoint chốt trên dataset cục bộ')
    evaluate_command.add_argument('--split',choices=('val','test'),default='test')
    evaluate_command.add_argument('--output',type=Path,required=True)
    evaluate_command.add_argument('--save-images',action='store_true')
    train=commands.add_parser('train',help='Chạy công thức 6 epoch vào một thư mục mới')
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--epochs',type=int,default=6)
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('Số luồng phải lớn hơn 0.')
    torch.set_num_threads(args.threads)
    try:
        if args.command=='gui':
            from .ui.desktop import launch
            launch(threads=args.threads,initial_image=args.input)
        elif args.command=='web':
            from .ui.local_web import launch
            launch(port=args.port,threads=args.threads,open_browser=not args.no_browser)
        elif args.command=='infer':
            infer(args)
        elif args.command=='evaluate':
            evaluate(args)
        elif args.command=='doctor':
            model,config=load_inference_model()
            print(json.dumps(dict(status='ok',project_root=str(ROOT),python=sys.executable,torch=str(torch.__version__),
                                  checkpoint=str(ROOT/config['model']),checkpoint_sha256=config['model_sha256'],
                                  parameters=sum(p.numel() for p in model.parameters()),dataset_required_for_photo=False,
                                  report=str(ROOT/config['evaluation_report'])),ensure_ascii=False,indent=2))
        else:
            from kind_enhance.training.curriculum import run_training
            run_training(args)
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f'Lỗi: {exc}\n')


if __name__=='__main__':main()

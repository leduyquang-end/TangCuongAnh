"""Chạy file này để mở giao diện tăng sáng ảnh tại localhost."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tăng sáng ảnh bằng checkpoint đã chốt.")
    parser.add_argument("--port", type=int, default=7860, help="Cổng localhost, mặc định 7860")
    parser.add_argument("--threads", type=int, default=6, help="Số luồng CPU, mặc định 6")
    parser.add_argument("--no-browser", action="store_true", help="Chạy server mà không mở trình duyệt")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Cổng phải nằm trong khoảng 1–65535.")
    if args.threads < 1:
        parser.error("Số luồng CPU phải lớn hơn 0.")
    try:
        from kind_enhance.ui.local_web import launch
        launch(port=args.port, threads=args.threads, open_browser=not args.no_browser)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Không chạy được ứng dụng: {exc}", file=sys.stderr)
        print("Chọn Python .venv/Scripts/python.exe của dự án; xem README.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

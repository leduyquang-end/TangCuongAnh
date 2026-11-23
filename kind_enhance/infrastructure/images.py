"""Read user photos consistently and save RGB PNGs without replacing originals."""
from pathlib import Path
import os
import shutil
import tempfile
from PIL import Image,ImageOps,UnidentifiedImageError


class ImageInputError(ValueError):
    pass


def read_photo(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy ảnh: {path}')
    try:
        with Image.open(path) as opened:
            oriented=ImageOps.exif_transpose(opened)
            if oriented.mode in ('RGBA','LA') or (oriented.mode=='P' and 'transparency' in oriented.info):
                rgba=oriented.convert('RGBA')
                background=Image.new('RGBA',rgba.size,'white')
                background.alpha_composite(rgba)
                result=background.convert('RGB')
            else:
                result=oriented.convert('RGB')
            result.load()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ImageInputError('Tệp không phải ảnh hợp lệ hoặc không thể đọc được.') from exc
    if min(result.size)<2:
        raise ImageInputError('Ảnh cần có ít nhất 2 pixel ở mỗi chiều.')
    return result


def prepare_photo(image,max_side=1280):
    if max_side!=0 and max_side<32:
        raise ValueError('Cạnh tối đa phải từ 32 pixel, hoặc 0 để giữ kích thước gốc.')
    prepared=image.copy()
    if max_side and max(prepared.size)>max_side:
        prepared.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    if min(prepared.size)<2:
        raise ImageInputError('Ảnh quá dài/hẹp để xử lý ở kích thước đã chọn.')
    return prepared


def save_png(image,path,overwrite=False):
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!='.png':
        raise ValueError('Ảnh kết quả phải có đuôi .png.')
    if path.exists() and not overwrite:
        raise FileExistsError(f'Tệp đã tồn tại: {path.name}. Chọn tên khác hoặc dùng --overwrite.')
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.image-',suffix='.png',dir=path.parent)
    os.close(descriptor)
    try:
        image.convert('RGB').save(temporary,format='PNG',compress_level=2)
        if path.exists() and not overwrite:
            raise FileExistsError(f'Tệp đã tồn tại: {path.name}')
        if overwrite:
            os.replace(temporary,path)
        else:
            created=False
            try:
                with path.open('xb') as destination:
                    created=True
                    with open(temporary,'rb') as source:
                        shutil.copyfileobj(source,destination)
            except Exception:
                if created:path.unlink(missing_ok=True)
                raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path

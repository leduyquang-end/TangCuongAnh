import csv
import torch
from PIL import Image,ImageDraw

STAGES = (
    ("01_input_low", "Anh toi | Input"),
    ("02_reflectance_R", "Phan xa | R"),
    ("03_illumination_L", "Chieu sang | L"),
    ("04_reconstruction_RxL", "Tai tao | R x L"),
    ("05_restored_reflectance", "Phan xa khoi phuc | R'"),
    ("06_adjusted_illumination", "Anh sang da chinh | L'"),
    ("07_enhanced_output", "Ket qua | R' x L'"),
    ("08_normal_reference", "Anh du sang | GT"),
)

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def pixels(image):
    if image.ndim == 4:
        image = image.squeeze(0)
    return (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def save_image(path, image):
    array = pixels(image)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path, compress_level=2)

def contact_sheet(folder, heading):
    sheet = Image.new("RGB", (1248, 496), "#f4f6fb")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), heading, fill="#152844")
    for index, (filename, label) in enumerate(STAGES):
        column, row = index % 4, index // 4
        left, top = 20 + 306 * column, 34 + 228 * row
        with Image.open(folder / f"{filename}.png") as picture:
            picture.thumbnail((288, 192))
            sheet.paste(picture.convert("RGB"), (left, top + 22))
        draw.text((left, top), label, fill="#152844")
    sheet.save(folder / "00_pipeline.png", compress_level=2)

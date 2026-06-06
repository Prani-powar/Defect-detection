import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import RAW_DATA_DIR


OUTPUT_DIR = RAW_DATA_DIR / "not_fruit" / "starter_negatives"
IMAGE_COUNT = 800
IMAGE_SIZE = 224
RANDOM_SEED = 71


def random_color(low: int = 0, high: int = 255) -> tuple[int, int, int]:
    return tuple(random.randint(low, high) for _ in range(3))


def draw_table_surface(draw: ImageDraw.ImageDraw) -> None:
    base = random_color(25, 210)
    for y in range(0, IMAGE_SIZE, random.randint(6, 18)):
        shade = tuple(max(0, min(255, channel + random.randint(-18, 18))) for channel in base)
        draw.rectangle((0, y, IMAGE_SIZE, y + random.randint(3, 12)), fill=shade)


def draw_grid_object(draw: ImageDraw.ImageDraw) -> None:
    fill = random_color(30, 230)
    outline = random_color(0, 80)
    x1 = random.randint(10, 70)
    y1 = random.randint(10, 80)
    x2 = random.randint(145, 215)
    y2 = random.randint(145, 215)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=random.randint(4, 18), fill=fill, outline=outline, width=3)
    step = random.randint(14, 28)
    for x in range(x1 + step, x2, step):
        draw.line((x, y1, x, y2), fill=outline, width=1)
    for y in range(y1 + step, y2, step):
        draw.line((x1, y, x2, y), fill=outline, width=1)


def draw_label_or_card(draw: ImageDraw.ImageDraw) -> None:
    x1 = random.randint(12, 55)
    y1 = random.randint(20, 90)
    x2 = random.randint(155, 215)
    y2 = random.randint(125, 205)
    draw.rectangle((x1, y1, x2, y2), fill=random_color(180, 255), outline=random_color(0, 70), width=2)
    for y in range(y1 + 18, y2 - 8, random.randint(12, 18)):
        draw.line((x1 + 12, y, x2 - 12, y), fill=random_color(20, 120), width=2)
    if random.random() < 0.5:
        draw.rectangle((x1 + 12, y1 + 12, x1 + 46, y1 + 46), fill=random_color(0, 120))


def draw_random_shapes(draw: ImageDraw.ImageDraw) -> None:
    for _ in range(random.randint(4, 12)):
        x1 = random.randint(0, IMAGE_SIZE - 40)
        y1 = random.randint(0, IMAGE_SIZE - 40)
        x2 = x1 + random.randint(18, 95)
        y2 = y1 + random.randint(18, 95)
        fill = random_color(10, 245)
        if random.random() < 0.5:
            draw.rectangle((x1, y1, x2, y2), fill=fill)
        else:
            draw.ellipse((x1, y1, x2, y2), fill=fill)


def make_image(index: int) -> Image.Image:
    background = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), random_color(15, 230))
    draw = ImageDraw.Draw(background)

    pattern = index % 5
    if pattern == 0:
        draw_table_surface(draw)
        draw_grid_object(draw)
    elif pattern == 1:
        draw_table_surface(draw)
        draw_label_or_card(draw)
    elif pattern == 2:
        draw_random_shapes(draw)
    elif pattern == 3:
        for _ in range(26):
            x = random.randint(0, IMAGE_SIZE)
            color = random_color(0, 255)
            draw.line((x, 0, random.randint(0, IMAGE_SIZE), IMAGE_SIZE), fill=color, width=random.randint(1, 5))
    else:
        draw_table_surface(draw)
        draw_random_shapes(draw)

    if random.random() < 0.45:
        background = background.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.7)))
    if random.random() < 0.35:
        background = background.rotate(random.uniform(-12, 12), resample=Image.Resampling.BILINEAR, fillcolor=random_color())
    return background


def main() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_DIR.glob("starter_not_fruit_*.jpg"))
    if len(existing) >= IMAGE_COUNT:
        print(f"Starter not_fruit images already exist: {len(existing)}")
        return

    for index in range(IMAGE_COUNT):
        image = make_image(index)
        image.save(OUTPUT_DIR / f"starter_not_fruit_{index:04d}.jpg", quality=88)
    print(f"Created {IMAGE_COUNT} starter not_fruit images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

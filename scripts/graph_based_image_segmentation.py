import os
from pathlib import Path
from utils import read_image, show_image, scale_image


def do():
    pass


if __name__ == "__main__":
    print(f"graph based image segmentation")
    print("testing only")

    base_path: Path = Path("/home/rajatshr/PycharmProjects/py-subdivision/test_images/")
    test_image: Path = base_path / "building.jpg"
    scale_factor: float = 0.15

    img = read_image(test_image)
    img = scale_image(img, scale=scale_factor)
    show_image(img)

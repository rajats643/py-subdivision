from pathlib import Path
import constants

import numpy as np
import cv2 as cv


def scale_image(
    image: np.ndarray, scale: float, interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    try:
        (h, w) = image.shape[:2]
        new_height, new_width = int(h * scale), int(w * scale)
        resized_image = cv.resize(
            image, (new_width, new_height), interpolation=interpolation
        )
    except (cv.error, ValueError, Exception) as e:
        raise RuntimeError(f"failed to scale image: {e}") from e
    return resized_image if resized_image.size > 0 else image


def gaussian_blur(image: np.ndarray, sigma: float = 0.8, k: int = 3) -> np.ndarray:
    try:
        result: np.ndarray = cv.GaussianBlur(src=image, ksize=[k, k], sigmaX=sigma)
    except (cv.error, Exception) as e:
        raise RuntimeError("failed to apply gaussian blur") from e
    return result


def median_blur(image: np.ndarray, k: int = 3) -> np.ndarray:
    try:
        result: np.ndarray = cv.medianBlur(src=image, ksize=k)
    except (cv.error, Exception) as e:
        raise RuntimeError("failed to apply median blur") from e
    return result


def show_image(image: np.ndarray, name: str = "image") -> None:
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyWindow(name)


def read_image(image_path: Path) -> np.ndarray:
    try:
        image: np.ndarray = cv.imread(str(image_path))
    except (FileNotFoundError, cv.error, Exception) as e:
        raise RuntimeError("failure during image read") from e
    return image


def write_image(image: np.ndarray, path: Path) -> None:
    try:
        cv.imwrite(str(path), image)
    except (cv.error, Exception) as e:
        raise RuntimeError(f"failure during image write: {e}") from e


if __name__ == "__main__":
    print("utility functions for image segmentation")
    print(f"using: \n open-cv:{cv.__version__}\n numpy {np.__version__}")

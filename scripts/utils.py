from pathlib import Path
import numpy as np
import cv2 as cv
import logging.config


def scale_image(
    image: np.ndarray, scale: float, interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    try:
        (h, w) = image.shape[:2]
        new_height, new_width = int(h * scale), int(w * scale)
        resized_image = cv.resize(
            image, (new_width, new_height), interpolation=interpolation
        )
        logger.debug(f"image scaled by {scale}, new shape: {resized_image.shape}")
    except (cv.error, ValueError, Exception) as e:
        raise RuntimeError("failed to scale image") from e
    return resized_image if resized_image.size > 0 else image


def show_image(image: np.ndarray, name: str = "image") -> None:
    logger: logging.Logger = logging.getLogger(__name__)
    logger.debug(f"showing image: {name}")
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyWindow(name)


def read_image(image_path: Path) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    logger.debug(f"reading image: {image_path}")
    try:
        image: np.ndarray = cv.imread(str(image_path))
        logger.debug(f"image shape: {image.shape}")
    except (FileNotFoundError, cv.error, Exception) as e:
        raise RuntimeError("failure during image read") from e
    return image


if __name__ == "__main__":
    print("utility functions for image segmentation")
    print(f"using: \n open-cv:{cv.__version__}\n numpy {np.__version__}")

from pathlib import Path
import numpy as np
import cv2 as cv
import logging.config


def setup_logging(debug_filename: Path, verbose: bool = False):
    handlers: list = ["console", "file"] if verbose else []
    log_config: dict = {
        "version": 1,
        "formatters": {
            "short": {
                "format": "%(funcName)s | %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": logging.INFO,
                "formatter": "short",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": logging.DEBUG,
                "formatter": "short",
                "filename": str(debug_filename),
                "mode": "w",
            },
        },
        "loggers": {
            "": {
                "handlers": handlers,
                "level": "DEBUG",
            },
        },
    }
    logging.config.dictConfig(log_config)


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


def gaussian_blur(image: np.ndarray, sigma: float = 0.8, k: int = 3) -> np.ndarray:
    logger: logging.Logger = logging.getLogger(__name__)
    try:
        result: np.ndarray = cv.GaussianBlur(src=image, ksize=[k, k], sigmaX=sigma)
        logger.debug(f"applied gaussian blur with sigma: {sigma}")
    except (cv.error, Exception) as e:
        raise RuntimeError("failed to apply gaussian blur") from e
    return result


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

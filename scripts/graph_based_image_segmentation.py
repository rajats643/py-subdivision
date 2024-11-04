import os
from pathlib import Path
from utils import read_image, show_image, scale_image
import logging.config
from random import choice


def setup_logging():
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
                "level": logging.DEBUG,
                "formatter": "short",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": "DEBUG",
            },
        },
    }
    logging.config.dictConfig(log_config)


if __name__ == "__main__":
    """
    graph based segmentation - testing only
    notes:
    """

    # setup params
    base_path: Path = Path(os.path.dirname(os.path.dirname(__file__)))
    test_images: list = ["building", "camera", "clothes", "island", "ski"]
    test_image: str = choice(test_images)
    test_image_path: Path = base_path / "test_images" / f"{test_image}.jpg"
    scale_factor: float = 0.15

    # setup logging
    setup_logging()
    logger: logging.Logger = logging.getLogger(__name__)
    logger.debug("setup logger")

    # operations
    img = read_image(test_image_path)
    img = scale_image(img, scale=scale_factor)
    show_image(img, test_image)

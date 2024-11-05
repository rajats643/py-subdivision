import os
from pathlib import Path
from typing import Dict, Set, Tuple, List

import numpy as np

from utils import read_image, show_image, scale_image
import logging.config
from random import choice
from collections import defaultdict


def get_adjacent(x: int, y: int, image: np.ndarray) -> List[Tuple[int, int]]:
    (h, w) = image.shape[:2]
    result: List[Tuple[int, int]] = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (0 <= (x + i) < h) or not (
                0 <= (y + j) < w
            ):  # if either coord out of bounds then exit
                continue
            result.append((x + i, y + j))
    return result


class ImageGraph:
    def __init__(self):
        self.G: Dict[int, Set[Tuple[int, float]]] = defaultdict(set)
        # G = {vertex: {(endpoint, weight), (endpoint, weight}, ...}

    def _add_vertex(self, vertex: int) -> None:
        if vertex not in self.G:
            self.G[vertex] = set()

    def _add_edge(self, vertex: int, endpoint: int, weight: float) -> None:
        if vertex not in self.G:
            self._add_vertex(vertex)
        if endpoint not in self.G[vertex]:
            self._add_vertex(endpoint)

        if not (vertex, weight) in self.G[endpoint]:
            self.G[vertex].add((endpoint, weight))

    def _build_graph_from_image(self, image: np.ndarray) -> np.ndarray:
        """image likely has three channels, we do only for one"""

        pass

    def __str__(self) -> str:
        result: str = ""
        for vertex, edges in self.G.items():
            result += f"{vertex}: {edges}\n"
        return result


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

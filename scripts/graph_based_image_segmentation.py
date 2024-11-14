import os
from pathlib import Path
from typing import Dict, Set, Tuple, List

import numpy as np

from utils import setup_logging, read_image, show_image, scale_image, gaussian_blur
import logging.config
from random import choice
from collections import defaultdict
from time import perf_counter


def get_adjacent_coords(x: int, y: int, image: np.ndarray) -> List[Tuple[int, int]]:
    (h, w) = image.shape[:2]
    result: List[Tuple[int, int]] = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            x_out_of_bounds: bool = not (0 <= (x + i) < h)
            y_out_of_bounds: bool = not (0 <= (y + j) < w)
            same_element: bool = not (i or j)
            if x_out_of_bounds or y_out_of_bounds or same_element:
                continue
            result.append((x + i, y + j))
    return result


# TODO: calculate mst prims and kruskal
# TODO: calculate cuts and regions
# TODO: merge regions
# TODO: segment images
# TODO: write tests
# TODO: remove dependency on openCV


def calculate_weights(x: int, y: int, image: np.ndarray) -> List[Tuple[int, int]]:
    """calculate weights of a point with respect to all adjacent points"""
    adjacent_coords: List[Tuple[int, int]] = get_adjacent_coords(x, y, image)
    weights: List[Tuple[int, int]] = []
    for px, py in adjacent_coords:
        weight: int = abs(int(image[px][py]) - int(image[x][y]))
        endpoint: int = (image.shape[0] * px) + py
        weights.append((endpoint, weight))
    return weights


class ImageComponent:

    def __init__(self):
        self.component: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

    def _add_vertex(self, vertex: int) -> None:
        if vertex not in self.component:
            self.component[vertex] = set()

    def _add_edge(self, vertex: int, endpoint: int, weight: int) -> None:
        if vertex not in self.component:
            self._add_vertex(vertex)
        if endpoint not in self.component[vertex]:
            self._add_vertex(endpoint)

        if not (vertex, weight) in self.component[endpoint]:
            self.component[vertex].add((endpoint, weight))


class ImageGraph:
    """build a graph from a single channel image"""

    def __init__(self, image: np.ndarray):
        # G = {vertex: {(endpoint, weight), (endpoint, weight),...}, ...}
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be np.ndarray")
        if len(image.shape) != 2:
            raise TypeError("image must be single channel 2d array")
        self.G: ImageComponent = ImageComponent()
        self._build_graph_from_image(image)

    def _add_vertex(self, vertex: int) -> None:
        if vertex not in self.G.component:
            self.G.component[vertex] = set()

    def _add_edge(self, vertex: int, endpoint: int, weight: int) -> None:
        if vertex not in self.G.component:
            self._add_vertex(vertex)
        if endpoint not in self.G.component[vertex]:
            self._add_vertex(endpoint)

        if not (vertex, weight) in self.G.component[endpoint]:
            self.G.component[vertex].add((endpoint, weight))

    def _build_graph_from_image(self, image: np.ndarray) -> None:
        """expecting single channel image"""
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                vertex: int = (image.shape[0] * x) + y
                connections: List[Tuple[int, int]] = calculate_weights(x, y, image)
                self._add_vertex(vertex)
                for endpoint, weight in connections:
                    self._add_edge(vertex, endpoint, weight)

    def __str__(self) -> str:
        result: str = ""
        for vertex, edges in self.G.component.items():
            result += f"{vertex}: {edges}\n"
        return result


def get_test_images(
    scale_factor: float = 0.1, channel: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    # setup params
    base_path: Path = Path(os.path.dirname(os.path.dirname(__file__)))
    test_images: list = ["building", "camera", "clothes", "island", "ski"]
    test_image_name: str = choice(test_images)
    test_image_path: Path = base_path / "test_images" / f"{test_image_name}.jpg"

    # operations
    image = read_image(test_image_path)
    image = scale_image(image, scale=scale_factor)
    image = gaussian_blur(image)

    # graph testing
    single_channel: np.ndarray = np.array(image[:, :, channel])
    logger.info(f"fetching image: {test_image_name}  img: {image.shape}")
    return image, single_channel


def runtime_test(n: int = 5, sf: float = 0.1) -> None:
    total: float = 0.0
    logger.info(f"running init runtime test ({n})")
    for _ in range(n):
        _, single = get_test_images(scale_factor=sf)
        start = perf_counter()
        _ = ImageGraph(single)
        result = perf_counter() - start
        logger.debug(f" init time: {result:.3f} seconds")
        total += result

    logger.info(f"total init time: {total:.3f} seconds")
    logger.info(f"average init time: {(total/n):.3f} seconds")


if __name__ == "__main__":
    """
    graph based segmentation - testing only
    notes:
    """
    verbose = True

    # setup logging
    setup_logging(debug_filename=Path("./graph.log"), verbose=verbose)
    logger: logging.Logger = logging.getLogger(__name__)
    logger.info("setup logger")

    # operations
    runtime_test(n=4, sf=0.1)
    logger.info("end of test")

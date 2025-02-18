# --------------------------- #
#   IMPORTS     #
# --------------------------- #

# built-in packages
import os
import logging.config
from pathlib import Path
from typing import Dict, Set, Tuple, List, Callable
from random import choice, randint
from time import perf_counter
import heapq
from workshop import FHSolver

from numpy.ma.core import absolute

# internal modules
import utils

# external modules
import numpy as np

# --------------------------- #
#   CLASSES     #
# --------------------------- #

# --------------------------- #
#   FUNCTIONS   #
# --------------------------- #


# --------------------------- #
#   WEIGHT FUNCTIONS    #
# --------------------------- #


# --------------------------- #
#   TEMP FUNCTIONS  #
# --------------------------- #


def get_test_image(
    scale_factor: float = 0.1, channel: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    # setup params
    base_path: Path = Path(os.path.dirname(os.path.dirname(__file__)))
    # test_images: list = ["building", "camera", "clothes", "island", "ski"]
    test_images: list = ["clothes"]
    test_image_name: str = choice(test_images)
    test_image_path: Path = base_path / "test_images" / f"{test_image_name}.jpg"

    # operations
    image = utils.read_image(test_image_path)
    image = utils.scale_image(image, scale=scale_factor)
    image = utils.gaussian_blur(image)

    # graph testing
    single_channel: np.ndarray = np.array(image[:, :, channel])
    logger.info(f"fetching image: {test_image_name}  img: {image.shape}")
    return image, single_channel


def get_adjacent_coords(x: int, y: int, image: np.ndarray) -> List[Tuple[int, int]]:
    # find valid adjacent coords for x,y in image
    h, w = image.shape
    result: List[Tuple[int, int]] = []

    for i in range(-1, 2):
        for j in range(-1, 2):

            x_out_of_bound: bool = not (0 <= (x + i) < h)
            y_out_of_bound: bool = not (0 <= (y + j) < w)
            same_element: bool = (i == 0) and (j == 0)

            if x_out_of_bound or y_out_of_bound or same_element:
                continue
            else:
                result.append((x + i, y + j))

    return result


def get_edges(image: np.ndarray) -> List[Tuple[int, int, float]]:
    h, w = image.shape
    global_set = set()
    result = []

    for i in range(h):
        for j in range(w):
            adj_list: List[Tuple[int, int]] = get_adjacent_coords(i, j, image)
            for adj in adj_list:
                weight: float = abs(float(image[i][j]) - float(image[adj[0]][adj[1]]))
                edge = (i * w + j, adj[0] * w + adj[1], weight)
                if edge not in global_set:
                    result.append(edge)
                    global_set.add(edge)
    return result


def runtime_test():
    image, single = get_test_image(scale_factor=0.15, channel=0)
    h, w = single.shape
    # utils.show_image(image)
    utils.show_image(single)
    edges = get_edges(single)
    logger.info(f"edges: {len(edges)}")
    start = perf_counter()
    edges = sorted(edges, key=lambda x: x[2])
    logger.info(f"edges sorted: {(perf_counter() - start):.2f} seconds")
    start = perf_counter()
    s = FHSolver(h, w, 15)

    for edge in edges:
        s.union(*edge)
    logger.info(f"fh segmentation: {(perf_counter() - start):.2f} seconds")

    for i in range(h * w):
        s.find(s.parent[i])

    # for i in range(h):
    #     for j in range(w):
    #         print(s.parent[i * w + j], end=" ")
    #     print()

    for i in range(h):
        for j in range(w):
            value = s.parent[i * w + j]
            new_value = single[value // w][value % w]
            single[i][j] = new_value

    utils.show_image(single)
    logger.info(f"number of components: {len(set(s.parent))}")


# --------------------------- #
#   MAIN    #
# --------------------------- #

if __name__ == "__main__":
    """
    graph based segmentation - testing only
    notes:
    """
    verbose = True

    # setup logging
    utils.setup_logging(debug_filename=Path("./graph.log"), verbose=verbose)
    logger: logging.Logger = logging.getLogger(__name__)
    logger.info("setup logger")

    # operations
    start_time = perf_counter()
    runtime_test()
    end_time = perf_counter()

    logger.info(f"runtime: {(end_time-start_time):.2f} seconds")
    logger.info("end of test")

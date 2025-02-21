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
import multiprocessing as mp

# internal modules
import utils

# external modules
import numpy as np

# --------------------------- #
#   CLASSES     #
# --------------------------- #


class FHSolver:

    def __init__(self, h: int, w: int, k: int):
        self.h: int = h
        self.w: int = w
        self.k: int = k
        self.parent: List[int] = [i for i in range(w * h)]
        self.internal_difference: List[float] = [0 for _ in range(w * h)]
        self.rank: List[int] = [1 for _ in range(w * h)]

    def find(self, x: int) -> int:
        """find the representative of this point"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int, weight: float) -> None:
        parent_x = self.find(x)
        parent_y = self.find(y)

        if parent_x == parent_y:
            return (
                None  # the edge between x,y forms a loop so we don't want to include it
            )

        mint_x: float = self.internal_difference[parent_x]
        mint_y: float = self.internal_difference[parent_y]
        mint: float = min(
            mint_x + (self.k / self.rank[parent_x]),
            mint_y + (self.k / self.rank[parent_y]),
        )
        if mint < weight:
            return None  # the edge doesn't meet the conditions of the FH predicate

        if self.rank[parent_x] < self.rank[parent_y]:
            self.parent[parent_x] = parent_y
        elif self.rank[parent_x] > self.rank[parent_y]:
            self.parent[parent_y] = parent_x
        else:
            self.parent[parent_y] = parent_x
            self.rank[parent_x] += 1

        max_edge_weight: float = max(
            self.internal_difference[x], self.internal_difference[y], weight
        )
        self.internal_difference[x] = max_edge_weight
        self.internal_difference[y] = max_edge_weight

    def __str__(self):
        result: str = ""
        result += f"Parent: {self.parent}\n"
        result += f"Rank: {self.rank}\n"
        result += f"Internal difference: {self.internal_difference}\n"
        return result


# --------------------------- #
#   FUNCTIONS   #
# --------------------------- #


def timeit(start: float, s: str = "") -> float:
    result = perf_counter() - start
    logger.info(f"{s} execution time: {result:.3f}")
    return result


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
    test_images: list = ["building", "camera", "clothes", "island", "ski"]
    # test_images: list = ["clothes"]
    # test_image_name: str = choice(test_images)
    for test_image_name in test_images:
        test_image_path: Path = base_path / "test_images" / f"{test_image_name}.jpg"

        # operations
        image = utils.read_image(test_image_path)
        image = utils.scale_image(image, scale=scale_factor)
        image = utils.gaussian_blur(image)

        # graph testing
        single_channel: np.ndarray = np.array(image[:, :, channel])
        logger.info(f"fetching image: {test_image_name}  img: {image.shape}")
        yield image, single_channel


def get_edges(image: np.ndarray) -> List[Tuple[int, int, float]]:
    h, w = image.shape
    result = []

    def form_edge(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int, float]:
        return (
            a[0] * w + a[1],
            b[0] * w + b[1],
            abs(float(image[a[0]][a[1]]) - float(image[b[0]][b[1]])),
        )

    for i in range(h - 1):
        for j in range(w - 1):
            current_point: Tuple[int, int] = (i, j)
            right_point: Tuple[int, int] = (i, j + 1)
            bottom_point: Tuple[int, int] = (i + 1, j)
            result.append(form_edge(current_point, right_point))
            result.append(form_edge(current_point, bottom_point))

    for j in range(w - 1):
        current_point: Tuple[int, int] = (h - 1, j)
        right_point: Tuple[int, int] = (h - 1, j + 1)
        result.append(form_edge(current_point, right_point))

    for i in range(h - 1):
        current_point: Tuple[int, int] = (i, w - 1)
        bottom_point: Tuple[int, int] = (i + 1, w - 1)
        result.append(form_edge(current_point, bottom_point))

    return result


def runtime_test(image, single, k=10):
    h, w = single.shape
    # utils.show_image(image)
    # utils.show_image(single)

    main_start = perf_counter()

    start = perf_counter()
    edges = get_edges(single)
    timeit(start, "get_edges")
    logger.info(f"edges: {len(edges)}")
    logger.info(f"expected edges: {(2 * h * w) - (h + w)}")

    start = perf_counter()
    edges = sorted(edges, key=lambda x: x[2])
    # logger.info(f"edges sorted: {(perf_counter() - start):.2f} seconds")
    timeit(start, "sorting")

    s = FHSolver(h, w, k)

    start = perf_counter()
    for edge in edges:
        s.union(*edge)
    timeit(start, "union / mst")

    start = perf_counter()
    for i in range(h * w):
        s.find(s.parent[i])
    timeit(start, "resolve parents")

    start = perf_counter()
    for i in range(h):
        for j in range(w):
            value = s.parent[i * w + j]
            new_value = single[value // w][value % w]
            single[i][j] = new_value
    timeit(start, "create mask")

    start = perf_counter()
    for i in range(h):
        for j in range(w):
            value = utils.GRAYSCALE_TO_RGB_MAP[single[i][j]]
            image[i][j][0] = value[0]
            image[i][j][1] = value[1]
            image[i][j][2] = value[2]
    timeit(start, "apply mask")
    timeit(main_start, "overall")

    # utils.show_image(image)
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

    for image, single in get_test_image(scale_factor=0.15, channel=0):
        utils.show_image(image)
        start_time = perf_counter()
        runtime_test(image, single, 30)
        runtime_test(image, np.array(image[:, :, 0]), 20)
        runtime_test(image, np.array(image[:, :, 0]), 10)
        end_time = perf_counter()
        utils.show_image(image)

    # logger.info(f"runtime: {(end_time-start_time):.2f} seconds")
    logger.info("end of test")

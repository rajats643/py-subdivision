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
from multiprocessing import shared_memory

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
        self.size: List[int] = [1 for _ in range(w * h)]

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
            mint_x + (self.k / self.size[parent_x]),
            mint_y + (self.k / self.size[parent_y]),
        )
        if mint < weight:
            return None  # the edge doesn't meet the conditions of the FH predicate

        if self.rank[parent_x] < self.rank[parent_y]:
            self.parent[parent_x] = parent_y
            self.size[parent_y] += self.size[parent_x]
        elif self.rank[parent_x] > self.rank[parent_y]:
            self.parent[parent_y] = parent_x
            self.size[parent_x] += self.size[parent_y]
        else:
            self.parent[parent_y] = parent_x
            self.rank[parent_x] += 1
            self.size[parent_x] += self.size[parent_y]

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
    # test_images: list = ["camera"]
    # test_image_name: str = choice(test_images)
    for test_image_name in test_images:
        test_image_path: Path = base_path / "test_images" / f"{test_image_name}.jpg"

        # operations
        image = utils.read_image(test_image_path)
        image = utils.scale_image(image, scale=scale_factor)
        image = utils.gaussian_blur(image, k=7)

        # graph testing
        single_channel: np.ndarray = np.array(image[:, :, channel])
        logger.info(f"fetching image: {test_image_name}  img: {image.shape}")
        yield image, single_channel


def get_edges(image: np.ndarray) -> List[Tuple[int, int, float]]:
    h, w = image.shape
    result: List[Tuple[int, int, float]] = []

    def form_edge(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int, float]:
        return (
            a[0] * w + a[1],
            b[0] * w + b[1],
            abs(
                float(image[a[0]][a[1]]) - float(image[b[0]][b[1]])
            ),  # absolute difference in pixel value
        )

    for i in range(h - 1):
        for j in range(w - 1):
            result.append(form_edge((i, j), (i, j + 1)))  # right edge
            result.append(form_edge((i, j), (i + 1, j)))  # bottom edge

    for j in range(w - 1):
        result.append(
            form_edge((h - 1, j), (h - 1, j + 1))
        )  # right edges on bottom row

    for i in range(h - 1):
        result.append(
            form_edge((i, w - 1), (i + 1, w - 1))
        )  # bottom edges on right column

    return result


def apply_mask(image: np.ndarray, mask: np.ndarray) -> None:
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            image[i][j] = utils.GRAYSCALE_TO_RGB_MAP[mask[i][j]]


def apply_triple_mask(image: np.ndarray, masks: List[np.ndarray]) -> None:
    h, w = masks[0].shape
    for i in range(h):
        for j in range(w):
            image[i][j][0] = masks[0][i][j]
            image[i][j][1] = masks[1][i][j]
            image[i][j][2] = masks[2][i][j]


def apply_segment_colors(image: np.ndarray) -> None:
    h, w, c = image.shape
    for i in range(h):
        for j in range(w):
            r, g, b = image[i][j]
            value = int((0.299 * r) + (0.587 * g) + (0.114 * b))
            image[i][j] = utils.GRAYSCALE_TO_RGB_MAP[value]


def get_fh_segmentation_mask(
    single_channel_image: np.ndarray,
    k: int = 10,
    result=None,
    channel: int = 0,
) -> None:

    # image dimensions
    h, w = single_channel_image.shape
    result_mask: np.ndarray = single_channel_image.copy()

    # get edges based on pixel values, sort by weight
    edges: List[Tuple[int, int, float]] = get_edges(single_channel_image)
    edges = sorted(edges, key=lambda x: x[2])

    # initialize solver
    solver: FHSolver = FHSolver(h, w, k)

    # form MST components
    for edge in edges:
        solver.union(*edge)

    # resolve parents
    for i in range(h * w):
        solver.find(solver.parent[i])

    # create segment mask
    for i in range(h):
        for j in range(w):
            value = solver.parent[i * w + j]
            result_mask[i][j] = single_channel_image[value // w][value % w]

    # set value in shared dictionary
    result.put((channel, result_mask))


def run_fh_rgb(image, k_scale=10, return_queue=None, grid_pos=0):
    # start_time = perf_counter()
    # utils.show_image(image)
    workers: List[mp.Process] = []
    result = mp.Queue()
    for channel in range(3):
        worker: mp.Process = mp.Process(
            target=get_fh_segmentation_mask,
            args=(
                image[:, :, channel],
                k_scale,
                result,
                channel,
            ),
        )
        workers.append(worker)
        worker.start()

    masks = [0, 0, 0]
    count = 0
    while count < 3:
        if not result.empty():
            temp = result.get()
            masks[temp[0]] = temp[1]
            count += 1

    for worker in workers:
        worker.join()

    result.close()
    apply_triple_mask(image, masks=masks)
    apply_segment_colors(image)
    return_queue.put((grid_pos, image))
    return_queue.close()
    # end_time = perf_counter()
    # logger.info(f" fh_time: {(end_time-start_time):.3f} seconds")


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

    scale_line: list = [1_500_000]
    result = mp.Queue()
    for test_image, sc in get_test_image(scale_factor=0.2, channel=0):
        # utils.show_image(test_image)
        h, w, c = test_image.shape
        grid: int = 1
        start_time = perf_counter()
        for scale in scale_line:
            workers: List[List[mp.Process]] = [
                [
                    mp.Process(
                        target=run_fh_rgb,
                        args=(
                            test_image[
                                i * (h // grid) : (i + 1) * (h // grid),
                                j * (w // grid) : (j + 1) * (w // grid),
                                :,
                            ],
                            scale,
                            result,
                            (i * grid) + j,
                        ),
                    )
                    for j in range(grid)
                ]
                for i in range(grid)
            ]

            for row in workers:
                for worker in row:
                    worker.start()

            count = 0
            while count < grid * grid:
                position, patch = result.get()
                i, j = divmod(position, grid)
                test_image[
                    i * (h // grid) : (i + 1) * (h // grid),
                    j * (w // grid) : (j + 1) * (w // grid),
                    :,
                ] = patch
                count += 1

            for row in workers:
                for worker in row:
                    worker.join()

        end_time = perf_counter()
        logger.info(f" fh_time: {(end_time-start_time):.3f} seconds")
        utils.show_image(test_image)

    # logger.info(f"runtime: {(end_time-start_time):.2f} seconds")
    logger.info("end of test")

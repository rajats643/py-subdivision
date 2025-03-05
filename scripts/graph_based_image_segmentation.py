# --------------------------- #
#   IMPORTS     #
# --------------------------- #

# built-in packages
import os
import logging.config
from email.mime import image
from pathlib import Path
from typing import Tuple, List
from time import perf_counter
import multiprocessing as mp

# internal modules
import utils

# external modules
import numpy as np

# --------------------------- #
#   CLASSES     #
# --------------------------- #


class FHSolver:
    """
    implements the FH segmentation method on an image using the union-find data structure
    """

    def __init__(self, h: int, w: int, k: int):
        # height, width of image
        self.h: int = h
        self.w: int = w

        # k is factor that determines the degree of segmentation
        # larger value -> larger segments
        self.k: int = k

        # parents (representatives) of each union
        self.parent: List[int] = [i for i in range(w * h)]

        # max weight in mst
        self.internal_difference: List[float] = [0 for _ in range(w * h)]

        # height of mst
        self.rank: List[int] = [1 for _ in range(w * h)]

        # number of nodes in mst
        self.size: List[int] = [1 for _ in range(w * h)]

    def find(self, x: int) -> int:
        """
        find the representative of x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int, weight: float) -> None:
        """
        combine the sets that x and y belong to, if they belong to different sets, and the weight between them exceeds M_INT
        """
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
        image = utils.gaussian_blur(image, k=3, sigma=0.8)

        # graph testing
        single_channel: np.ndarray = np.array(image[:, :, channel])
        logger.info(f"fetching image: {test_image_name}  img: {image.shape}")
        yield image, single_channel


def get_edges(image: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    return a list of edges formed from an image
    """
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
    """
    overwrite image using a mask to color segments
    """
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            image[i][j] = utils.GRAYSCALE_TO_RGB_MAP[mask[i][j]]


def apply_triple_mask(image: np.ndarray, masks: List[np.ndarray]) -> None:
    """
    overwrite image by applying RGB masks
    """
    h, w = masks[0].shape
    for i in range(h):
        for j in range(w):
            image[i][j][0] = masks[0][i][j]
            image[i][j][1] = masks[1][i][j]
            image[i][j][2] = masks[2][i][j]


def apply_segment_colors(image: np.ndarray) -> None:
    """
    overwrite image by converting RGB pixels -> grayscale mapping -> RGB segmentation colors
    """
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
    """
    perform FH segmentation algorithm on a single channel image
    """

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


def run_fh_rgb(image, k_scale=10):
    workers: List[mp.Process] = []
    result = mp.Queue()

    # get FH segmentation masks for each channel
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

    # retrieve results from result Queue
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

    scale_line: list = [15_00]
    result = mp.Queue()
    for test_image, sc in get_test_image(scale_factor=0.15, channel=0):
        # utils.show_image(test_image)
        h, w, c = test_image.shape
        """
        grid: int = 2
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
        """

        start_time = perf_counter()
        for scale in scale_line:
            run_fh_rgb(test_image, k_scale=scale)

        end_time = perf_counter()
        test_image = utils.median_blur(test_image, k=5)
        logger.info(f" fh_time: {(end_time-start_time):.3f} seconds")
        utils.show_image(test_image)

    # logger.info(f"runtime: {(end_time-start_time):.2f} seconds")
    logger.info("end of test")

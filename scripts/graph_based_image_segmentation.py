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

# internal modules
import utils

# external modules
import numpy as np

# --------------------------- #
#   CLASSES     #
# --------------------------- #


class Vertex:
    """a vertex class that contains the coordinates of a single vertex from the image"""

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"


class Edge:
    """an edge class that contains two vertices that make an edge and its weight"""

    def __init__(self, v1: Vertex, v2: Vertex):
        self.v1: Vertex = v1
        self.v2: Vertex = v2
        self.weight: float = 0

    def set_weight(
        self,
        image: np.ndarray,
        weight_function: Callable[[Vertex, Vertex, np.ndarray], float],
    ):
        self.weight = weight_function(self.v1, self.v2, image)

    def __lt__(self, other) -> bool:
        return self.weight < other.weight

    def __str__(self):
        return f"v1: {self.v1} v2: {self.v2} weight: {self.weight}"


class Component:
    """a component that defines an area of an image using its constituent edges"""

    def __init__(self, edges: Set[Edge]):
        self.edges: Set[Edge] = edges

    def edge_presence(self, edge: Edge) -> bool:
        return edge in self.edges


# --------------------------- #
#   FUNCTIONS   #
# --------------------------- #


def merge_components(a: Component, b: Component) -> Component:
    return Component(a.edges.union(b.edges))


# --------------------------- #
#   WEIGHT FUNCTIONS    #
# --------------------------- #


def absolute_intensity(v1: Vertex, v2: Vertex, image) -> float:
    return abs(image[v1.x][v1.y] - image[v2.x][v2.y])


# --------------------------- #
#   TEMP FUNCTIONS  #
# --------------------------- #


def get_test_images(
    scale_factor: float = 0.1, channel: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    # setup params
    base_path: Path = Path(os.path.dirname(os.path.dirname(__file__)))
    test_images: list = ["building", "camera", "clothes", "island", "ski"]
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


def get_list(n: int, image: np.ndarray) -> List[Edge]:
    result = []
    for _ in range(n):
        v1 = Vertex(randint(0, n), randint(0, n))
        v2 = Vertex(randint(0, n), randint(0, n))
        edge = Edge(
            v1,
            v2,
        )
        edge.set_weight(image, absolute_intensity)
        result.append(edge)

    return result


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
    # runtime_test(n=4, sf=0.1)
    sample = np.array([[x for x in range(11)] for _ in range(11)])

    list_edges = get_list(10, sample)
    for edge in list_edges:
        print(edge)

    print()

    for edge in sorted(list_edges):
        print(edge)

    print(list_edges[3] > list_edges[1])

    logger.info("end of test")

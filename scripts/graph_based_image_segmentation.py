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

from numpy.ma.core import absolute

# internal modules
import utils

# external modules
import numpy as np

# --------------------------- #
#   CLASSES     #
# --------------------------- #


class UnionFind:
    """an implementation of the union-find data structure"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[y_root] < self.rank[x_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1


class Vertex:
    """a vertex class that contains the coordinates of a single vertex from the image"""

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"

    def __hash__(self):
        return hash((self.x, self.y))


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

    def __hash__(self):
        return hash((self.v1, self.v2, self.weight))


class Component:
    """a component that defines an area of an image using its constituent edges"""

    def __init__(self, edges: Set[Edge]):
        self.edges: Set[Edge] = edges
        self.sorted_edges: List[Edge] = []
        self.mst: Set[Edge] = set()
        self.mst_vertices: Set[Vertex] = set()

    def __len__(self):
        return len(self.edges)

    def construct_mst(self):
        """use Kruskal's algorithm to find MST"""
        self.mst_vertices: Set[Vertex] = set()
        self.mst: Set[Edge] = set()
        self.sorted_edges = sorted(self.edges)
        for edge in self.sorted_edges:
            # check if edge forms a cycle
            if edge.v1 in self.mst_vertices and edge.v2 in self.mst_vertices:
                print("skip")
            else:
                self.mst.add(edge)
                self.mst_vertices.add(edge.v1)
                self.mst_vertices.add(edge.v2)

    def edge_presence(self, edge: Edge) -> bool:
        return edge in self.edges

    def set_weights(
        self,
        image: np.ndarray,
        weight_function: Callable[[Vertex, Vertex, np.ndarray], float],
    ):
        for edge in self.edges:
            edge.set_weight(image, weight_function)


"""  
Sort all the edges in non-decreasing order of their weight.
Pick the smallest edge. Check if it forms a cycle with the spanning tree formed so far. If the cycle is not formed, include this edge. Else, discard it.
Repeat step#2 until there are (V-1) edges in the spanning tree.

"""
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


def get_test_image(
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


def image_to_component(image: np.ndarray) -> Component:
    # image is a single channel
    h, w = image.shape
    edges: Set[Edge] = set()

    for i in range(h):
        for j in range(1, w):
            current_point: Vertex = Vertex(i, j)
            previous_point: Vertex = Vertex(i, j - 1)
            edges.add(Edge(current_point, previous_point))

    for j in range(w):
        for i in range(1, h):
            current_point: Vertex = Vertex(i, j)
            previous_point: Vertex = Vertex(i - 1, j)
            edges.add(Edge(current_point, previous_point))

    component = Component(edges)
    component.set_weights(image, absolute_intensity)
    return component


def runtime_test():
    image, single = get_test_image()
    h, w = single.shape
    logger.info(f"creating component")
    start = perf_counter()
    component = image_to_component(single)
    end = perf_counter()
    logger.info(f"created component: {(end - start):.2f} seconds")
    logger.info(f"component size: {len(component)}")
    logger.info(f"expected size: {(2*h*w) - (h+w)}")
    logger.info("constructing mst")
    start = perf_counter()
    component.construct_mst()
    end = perf_counter()
    logger.info(f"construct mst: {(end - start):.2f} seconds")
    logger.info(f"mst size: {len(component.mst)}")


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

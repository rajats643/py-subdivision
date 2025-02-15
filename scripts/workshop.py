import numpy as np
from typing import List, Tuple, Callable


class FHSolver:

    def __init__(self, h: int, w: int, edges: List[Tuple[int, int]] = None):
        self.edges: List[Tuple[int, int, float]] = edges if edges is not None else []
        self.h: int = h
        self.w: int = w
        self.uf_matrix: List[List[int]] = [
            [(i * w) + j for j in range(w)] for i in range(h)
        ]
        self.id_matrix: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]

    def set_edges(self, edges: List[Tuple[int, int, float]]):
        self.edges: List[Tuple[int, int, float]] = edges

    def find(self, i: int, j: int) -> int:
        parent: int = (i * self.w) + j
        if self.uf_matrix[i][j] == parent:
            return parent

        self.uf_matrix[i][j] = self.find(parent // 10, parent % 10)
        return self.uf_matrix[i][j]


def main():
    x = [
        (1, 2, 10),
        (2, 3, 20),
        (3, 9, 3),
        (3, 4, 40),
        (4, 1, 40),
        (4, 5, 80),
        (5, 9, 50),
        (9, 2, 40),
    ]
    s = FHSolver(5, 5)
    print(s.uf_matrix)
    print(s.id_matrix)


if __name__ == "__main__":
    main()

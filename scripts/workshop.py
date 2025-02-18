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
        self.id_matrix: List[List[float]] = [[0 for _ in range(w)] for _ in range(h)]
        self.rank: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]

    def set_edges(self, edges: List[Tuple[int, int, float]]):
        self.edges: List[Tuple[int, int, float]] = edges

    def find(self, i: int, j: int) -> int:
        parent: int = (i * self.w) + j
        if self.uf_matrix[i][j] == parent:
            return parent

        self.uf_matrix[i][j] = self.find(parent // 10, parent % 10)
        return self.uf_matrix[i][j]

    def union(self, point_one: Tuple[int, int], point_two: Tuple[int, int]) -> None:
        parent_one: int = self.find(point_one[0], point_one[1])
        parent_two: int = self.find(point_two[0], point_two[1])

        if parent_one == parent_two:
            return None  # the edge forms a loop, and we don't want to include it

        one: Tuple[int, int] = parent_one // 10, parent_one % 10
        two: Tuple[int, int] = parent_two // 10, parent_two % 10

        if self.rank[one[0]][one[1]] < self.rank[two[0]][two[1]]:
            self.uf_matrix[one[0]][one[1]] = parent_two
        elif self.rank[one[0]][one[1]] > self.rank[two[0]][two[1]]:
            self.uf_matrix[two[0]][two[1]] = parent_one
        else:
            self.uf_matrix[two[0]][two[1]] = parent_one
            self.rank[two[0]][two[1]] += 1

        max_edge_weight: float = max(
            self.id_matrix[one[0]][one[1]], self.id_matrix[two[0]][two[1]]
        )
        self.id_matrix[one[0]][one[1]] = max_edge_weight
        self.id_matrix[two[0]][two[1]] = max_edge_weight


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

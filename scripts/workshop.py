import numpy as np
from typing import List, Tuple, Callable
import time


class FHSolver:

    def __init__(self, h: int, w: int, k: int):
        self.h: int = h
        self.w: int = w
        self.k: int = k
        self.parent: List[int] = [i for i in range(w * h)]
        self.internal_difference: List[float] = [0 for _ in range(w * h)]
        self.rank: List[int] = [1 for _ in range(w * h)]

    def find(self, x: int) -> int:
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
    x = sorted(x, key=lambda x: x[2])
    s = FHSolver(3, 4, 10)
    start = time.perf_counter()

    print(s)
    for edge in x:
        s.union(*edge)
    print(s)
    print(f"Execution time: {(time.perf_counter() - start):.3f} seconds")


if __name__ == "__main__":
    main()

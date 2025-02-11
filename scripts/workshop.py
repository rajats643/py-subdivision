import os


class Graph:
    """an implementation of a graph containing union-find methods for MST"""

    def __init__(self, vertices: int):
        self.vertices: int = vertices
        self.graph: list = []
        self.parent: list = [i for i in range(vertices)]
        self.rank: list = [0] * vertices
        self.mst: list = []
        self.internal_difference: float = -1

    def add_edge(self, u: int, v: int, weight: float) -> None:
        self.graph.append((u, v, weight))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        x_root: int = self.find(x)
        y_root: int = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[y_root] < self.rank[x_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def build_mst(self):
        self.mst: list = []
        edge_counter: int = 0
        mst_counter: int = 0

        self.graph = sorted(self.graph, key=lambda x: x[2])  # sort edges by weight
        while mst_counter < self.vertices - 1:
            u, v, w = self.graph[edge_counter]
            edge_counter += 1
            x: int = self.find(u)
            y: int = self.find(v)
            if x != y:
                mst_counter += 1
                self.mst.append((u, v, w))
                self.union(x, y)
        self.internal_difference = self.mst[-1][2]  # maximum weight in the MST


if __name__ == "__main__":
    print(os.name)

"""
Implementation of Minimum Spanning Tree
"""


class MinimumSpanningTree:
    """
    Minimum Spanning Tree class using 'Kruskal algorithm' to get distances

    A Minimum Spanning Tree (MTS) constructs a graph with vertices without
    loops using edges, such that the total weight of all edges is minimized.

    Reference:
        You can find Kruskal algorithm in the first 2 pages of the below paper.
        - A review of two decades of correlations, hierarchies, networks and clustering in financial markets
        (https://arxiv.org/pdf/1703.00485.pdf)

    """
    def __init__(self, vertices, edges):
        """
        Parameters
        ----------
            vertices: (list) : List of name of vertices.
                ex) ['AAPL', 'AMZN', 'XYL', 'GLW'...]
            edges : (list) : List of tuple which contain weight, name of vertexs.
                             The list should be arranged in ascending order by weight of each tuple.

        Examples
        --------
            >>>> vertices = ['AAPL', 'AMZN', 'XYL', 'GLW'...]
            >>>> edges = [(0.5738119476776615, 'XYL', 'GE'),
                     (0.6393385878075669, 'GLW', 'GE'),
                     (0.6748059132700998, 'NTRS', 'XYL'),
                     (0.7090918442572199, 'NTRS', 'GE'),
                     (0.7506079746487421, 'NTRS', 'GLW'),
                     (0.7592144174674299, 'GLW', 'XYL')
                     ...
                     ]

            >>>> mst = MinimumSpanningTree(vertices, edges)
            >>>> graph = mst.build()
            >>>> graph
            [(0.5738119476776615, 'XYL', 'GE'),
             (0.6393385878075669, 'GLW', 'GE'),
             (0.6748059132700998, 'NTRS', 'XYL'),
             (0.7665585346886722, 'TROW', 'GLW')]

        """
        self.vertices = vertices
        self.edges = edges
        self.parent = {}
        self.rank = {}
        self.init_table()
        self.__graph = []

    def init_table(self):
        """
        This initializes dictionaries used to build tree.
        """
        for vertex in self.vertices:
            self.parent[vertex] = vertex
            self.rank[vertex] = 0

    @property
    def graph(self):
        """
        Return the information of tree graph
        """
        return self.__graph

    def find_parent(self, vertex):
        """
        This method is used for finding parent(root) of vertex(node)

        Parameters
        ----------
            vertex: (string) : name of vertex

        Return
        -------
            parent of vertex: (string) : name of parent of vertex

        """
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find_parent(self.parent[vertex])

        return self.parent[vertex]

    def check_connected(self, vertex_1, vertex_2):
        """
        This method is used to check whether two vertices is connected
        If vertices are connected, it will return True, else Fasle

        Parameters
        ----------
            vertex_1: (string) : name of vertex
            vertex_2: (string) : name of vertex

        Return
        -------
            connected: (bool) : If vertices are connected, it will return True, else Fasle
        """
        parent1 = self.find_parent(vertex_1)
        parent2 = self.find_parent(vertex_2)

        return parent1 == parent2

    def union_parent(self, vertex_1, vertex_2):
        """
        This method is used to union parents of vertices.
        It unions based on the parent of 'vertex_1' vertex.
        Parameters
        ----------
            vertex_1: (string) : name of vertex
            vertex_2: (string) : name of vertex

        """
        parent1 = self.find_parent(vertex_1)
        parent2 = self.find_parent(vertex_2)

        if parent1 != parent2:
            self.parent[parent2] = parent1

    def build(self):
        """
        It builds the minimum spanning tree(MST) based on Kruskal's algorithm

        Note:
             If there are N vertices, this MST selects N-1 shortest edges(links) that span all the
             vertices without forming loops. We can finish build tree when there are N-1 edges in tree
             before looping all edges.
        """
        for edge in self.edges:
            _, vertex_1, vertex_2 = edge
            if not self.check_connected(vertex_1, vertex_2):
                self.union_parent(vertex_1, vertex_2)
                self.__graph.append(edge)

                if len(self.__graph) == len(self.vertices)-1:
                    break

        print(f"The building of tree is completed")
        return self.__graph

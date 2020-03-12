"""
Tests MST(Minimum Spanning Tree) class in mst.py
"""

import itertools
import unittest

from mst import MinimumSpanningTree


class TestMST(unittest.TestCase):
    """
    Tests MST(Minimum Spanning Tree) class
    """

    @classmethod
    def setUpClass(cls):
        # Setup graph information.
        cls.vertices = ['A', 'B', 'C', 'D', 'E']
        cls.edges = [
            [1, 'A', 'E'],
            [2, 'C', 'D'],
            [3, 'A', 'B'],
            [4, 'B', 'E'],
            [5, 'B', 'C'],
            [6, 'E', 'C'],
            [7, 'E', 'D']
        ]

        # This tree will be used for final test whether tree make a correct tree as a result.
        cls.mst = MinimumSpanningTree(cls.vertices, cls.edges)
        cls.mst.build()

    def test_find_parent_when_connected(self):
        """
        Test whether tree find root parent of vertex correctly when vertices are connected.
        """
        vertices = ['A', 'B', 'E']
        edges = [
            [1, 'A', 'E'],
            [3, 'A', 'B'],
        ]
        mst = MinimumSpanningTree(vertices, edges)
        mst.build()
        parent = list(mst.parent.values())

        # root parent should be 'A'
        self.assertListEqual(parent, ['A', 'A', 'A'])

    def test_find_parent_when_disconnected(self):
        """
        Test whether tree find root parent of vertex correctly when vertices are not fully connected.
        """
        vertices = ['A', 'E', 'C', 'D']
        edges = [
            [1, 'A', 'E'],
            [2, 'C', 'D'],
        ]
        mst = MinimumSpanningTree(vertices, edges)
        mst.build()
        parent = list(mst.parent.values())
        answer = ['A', 'A', 'C', 'C']
        self.assertListEqual(parent, answer)

    def test_check_connected_when_connected(self):
        """
        Test whether tree check root connection of tree correctly when vertices are connected.
        """
        vertices = ['A', 'B', 'E']
        edges = [
            [1, 'A', 'E'],
            [3, 'A', 'B'],
        ]
        mst = MinimumSpanningTree(vertices, edges)
        mst.build()

        for pairs in list(itertools.combinations(vertices, 2)):
            # All pairs should be connected
            vertex_1, vertex_2 = pairs
            is_connected = mst.check_connected(vertex_1, vertex_2)
            self.assertTrue(is_connected)

    def test_check_connected_when_disconnected(self):
        """
        Test whether tree check root connection of tree correctly vertices are not fully connected.
        """
        vertices = ['A', 'E', 'C', 'D']
        edges = [
            [1, 'A', 'E'],
            [2, 'C', 'D'],
        ]
        mst = MinimumSpanningTree(vertices, edges)
        mst.build()

        for pairs in list(itertools.combinations(vertices, 2)):
            # All pairs should be connected except {'A', 'E'} and {'C', 'D'}
            vertex_1, vertex_2 = pairs
            pairs_set = set(pairs)
            is_connected = mst.check_connected(vertex_1, vertex_2)
            if pairs_set == {'A', 'E'} or pairs_set == {'C', 'D'}:
                pass
            else:
                is_connected = not is_connected

            self.assertTrue(is_connected)

    def test_union_parent(self):
        """
        Test whether tree union parents of two vertices correctly
        """
        vertices = ['A', 'B', 'E']
        edges = [
            [1, 'A', 'E'],
            [3, 'A', 'B'],
        ]
        mst = MinimumSpanningTree(vertices, edges)
        mst.union_parent('A', 'B')

        # After union, parent of all vertices should be 'A' except vertex 'E'
        parent = list(mst.parent.values())
        self.assertListEqual(parent, ['A', 'A', 'E'])

    def test_the_number_of_edges(self):
        """
        Test the number of edges in Tree.
        When there are 'N' vertices, there should be 'N-1' edges.
        """
        num_vertices = len(self.mst.vertices)
        num_edges = len(self.mst.graph)

        self.assertEqual(num_vertices-1, num_edges)


if __name__ == '__main__':
    unittest.main()

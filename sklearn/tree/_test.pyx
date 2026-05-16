from collections import namedtuple
from libc.math cimport INFINITY

from ._honest_tree import HonestTree

from ._honesty cimport Honesty, HonestEnv, Views
from ._tree cimport BaseTree, Tree


Interval = namedtuple('Interval', ['lower', 'upper'])


cdef class TestNode():
    def __init__(self, bounds : [Interval], start_idx, n):
        self.bounds = bounds
        self.start_idx = start_idx
        self.n = n
    
    def valid(self, float32_t[:, :] X, intp_t[:] samples):
        for i in range(self.start_idx, self.start_idx + self.n):
            for j in range(len(self.bounds)):
                if X[samples[i]][j] < self.bounds[j].lower:
                    print("")
                    print(f"start_idx = {self.start_idx}")
                    print(f"n = {self.n}")
                    print(f"dimension = {j}")
                    print(f"X.shape = {X.shape}")
                    print(f"bounds = {self.bounds[j]}")
                    print(f"range = {[i for i in range(self.start_idx, self.start_idx + self.n)]}")
                    print(f"failed on {X[samples[i]][j]} < {self.bounds[j].lower}")
                    print(f"leaf feature values = {[ X[samples[ii]][j] for ii in range(self.start_idx, self.start_idx + self.n) ]}")
                    return False
                
                if X[samples[i]][j] > self.bounds[j].upper:
                    print("")
                    print(f"start_idx = {self.start_idx}")
                    print(f"n = {self.n}")
                    print(f"dimension = {j}")
                    print(f"X.shape = {X.shape}")
                    print(f"bounds = {self.bounds[j]}")
                    print(f"range = {[i for i in range(self.start_idx, self.start_idx + self.n)]}")
                    print(f"failed on {X[samples[i]][j]} > {self.bounds[j].upper}")
                    print(f"leaf feature values = {[ X[samples[ii]][j] for ii in range(self.start_idx, self.start_idx + self.n) ]}")
                    return False
        
        return True
    
    def to_dict(self):
        return {
            "bounds": self.bounds,
            "start_idx": self.start_idx,
            "n": self.n
        }


cdef class HonestyTester():
    def __init__(self, honest_tree: HonestTree):
        cdef Honesty honesty = honest_tree.honesty
        cdef Tree t = honest_tree.target_tree.tree_

        self.nodes = t.nodes
        self.intervals = honesty.env.tree
        self.X = honesty.views.X
        self.samples = honesty.views.samples


    #cdef struct Node:
    #    # Base storage structure for the nodes in a Tree object
    #
    #    intp_t left_child                    # id of the left child of the node
    #    intp_t right_child                   # id of the right child of the node
    #    intp_t feature                       # Feature used for splitting the node
    #    float64_t threshold                  # Threshold value at the node
    #    float64_t impurity                   # Impurity of the node (i.e., the value of the criterion)
    #    intp_t n_node_samples                # Number of samples at the node
    #    float64_t weighted_n_node_samples    # Weighted number of samples at the node
    #    unsigned char missing_go_to_left     # Whether features have missing values

    def get_invalid_nodes(self):
        return [
            n for n in self.to_cells()
            if not n.valid(self.X, self.samples)
        ]


    def to_cells(self, intp_t node_id = 0, bounds : [Interval] = None):
        cdef Node* node = &self.nodes[node_id]
        if bounds is None:
            bounds = [
                Interval(-INFINITY, INFINITY)
                for _ in range(self.X.shape[0])
            ]

        if node.feature < 0:
            return [
                TestNode(
                    bounds,
                    self.intervals[node_id].start_idx,
                    self.intervals[node_id].n
                )
            ]
        else:
            return self.to_cells(
                node.left_child,
                [
                    Interval(bounds[j].lower, node.threshold)
                    if j == node.feature
                    else bounds[j]
                    for j in range(len(bounds))
                ]
            ) + self.to_cells(
                node.right_child,
                [
                    Interval(node.threshold, bounds[j].upper)
                    if j == node.feature
                    else bounds[j]
                    for j in range(len(bounds))
                ]
            )

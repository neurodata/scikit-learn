# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#
# License: BSD 3 clause

# See _splitter.pyx for details.
cimport numpy as cnp

from libcpp.vector cimport vector

from ._criterion cimport BaseCriterion, Criterion
from ._tree cimport DOUBLE_t  # Type of y, sample_weight
from ._tree cimport DTYPE_t  # Type of X
from ._tree cimport INT32_t  # Signed 32 bit integer
from ._tree cimport SIZE_t  # Type for indices and counters
from ._tree cimport UINT32_t  # Unsigned 32 bit integer


cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
    #                      # i.e. count of samples below threshold for feature.
    #                      # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.
    double lower_bound     # Lower bound on value of both children for monotonicity
    double upper_bound     # Upper bound on value of both children for monotonicity
    unsigned char missing_go_to_left  # Controls if missing values go to the left node.
    SIZE_t n_missing       # Number of missing values for the feature being split on

cdef class BaseSplitter:
    """Abstract interface for splitter."""

    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    cdef public double min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t[::1] samples             # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t[::1] features            # Feature indices in X
    cdef SIZE_t[::1] constant_features   # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t[::1] feature_values     # temp. array holding feature values

    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    # Monotonicity constraints for each feature.
    # The encoding is as follows:
    #   -1: monotonic decrease
    #    0: no constraint
    #   +1: monotonic increase
    cdef const cnp.int8_t[:] monotonic_cst
    cdef bint with_monotonic_cst
    cdef const DOUBLE_t[:] sample_weight

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) except -1 nogil
    cdef int node_split(
        self,
        double impurity,   # Impurity of the node
        SplitRecord* split,
        SIZE_t* n_constant_features,
        double lower_bound,
        double upper_bound,
    ) except -1 nogil
    cdef void node_value(self, double* dest) noexcept nogil
    cdef void clip_node_value(
        self,
        double* dest,
        double lower_bound,
        double upper_bound
    ) noexcept nogil
    cdef double node_impurity(self) noexcept nogil
    cdef int pointer_size(self) noexcept nogil

cdef class Splitter(BaseSplitter):
    cdef public Criterion criterion      # Impurity criterion
    cdef const DOUBLE_t[:, ::1] y

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1

    cdef void node_samples(self, vector[vector[DOUBLE_t]]& dest) noexcept nogil

    # Methods that allow modifications to stopping conditions
    cdef bint check_presplit_conditions(
        self,
        SplitRecord current_split,
        SIZE_t n_missing,
        bint missing_go_to_left,
    ) noexcept nogil

    cdef bint check_postsplit_conditions(
        self
    ) noexcept nogil

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#          Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause
# SPDX-License-Identifier: BSD-3-Clause

from ..utils._typedefs cimport float32_t, float64_t, intp_t, int8_t, uint8_t, int32_t, uint32_t

# Constant to switch between algorithm non zero value extract algorithm
# in SparsePartitioner
cdef float32_t EXTRACT_NNZ_SWITCH = 0.1

# We introduce a different approach to the fused type for {Dense, Sparse}Partitioner.
# The main drawback of the fused type approach is that it seemed to require a
# proliferation of concrete Splitter types in order to accommodate holding ownership
# of each concrete type of Partitioner, hence the
# {Best, BestSparse, Random, RandomSparse}Splitter classes. This pattern generalizes
# to any class wishing to hold a concrete instance of Partitioner, which makes
# reusing the Partitioner code (as we wish to do for honesty and obliqueness) a
# fractal class-generating process.
#
# The alternative we introduce is the same pattern we use all over the place:
# function pointers. Assigning method implementations as function pointer values
# in init allows DensePartitioner and SparsePartitioner to be plain old subclasses
# of Partitioner, and there is no performance hit from virtual method lookup.
#
# Since we also seek to reuse Partitioner as its own module, we break it out into
# its own files.

# Introduce a fused-class to make it possible to share the split implementation
# between the dense and sparse cases in the node_split_best and node_split_random
# functions. The alternative would have been to use inheritance-based polymorphism
# but it would have resulted in a ~10% overall tree fitting performance
# degradation caused by the overhead frequent virtual method lookups.
#ctypedef fused Partitioner:
#    DensePartitioner
#    SparsePartitioner


ctypedef void (*InitNodeSplitFunction)(
    Partitioner partitioner, intp_t start, intp_t end
) noexcept nogil

ctypedef void (*SortSamplesAndFeatureValuesFunction)(
    Partitioner partitioner, intp_t current_feature
) noexcept nogil

ctypedef void (*FindMinMaxFunction)(
    Partitioner partitioner,
    intp_t current_feature,
    float32_t* min_feature_value_out,
    float32_t* max_feature_value_out,
) noexcept nogil

ctypedef void (*NextPFunction)(
    Partitioner partitioner, intp_t* p_prev, intp_t* p
) noexcept nogil

ctypedef intp_t (*PartitionSamplesFunction)(
    Partitioner partitioner, float64_t current_threshold
) noexcept nogil

ctypedef void (*PartitionSamplesFinalFunction)(
    Partitioner partitioner,
    intp_t best_pos,
    float64_t best_threshold,
    intp_t best_feature,
    intp_t best_n_missing,
) noexcept nogil


cdef class Partitioner:
    cdef:
        intp_t[::1] samples
        float32_t[::1] feature_values
        intp_t start
        intp_t end
        intp_t n_missing
        const uint8_t[::1] missing_values_in_feature_mask

        inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil
        inline void sort_samples_and_feature_values(
            self,
            intp_t current_feature
        ) noexcept nogil
        inline void find_min_max(
            self,
            intp_t current_feature,
            float32_t* min_feature_value_out,
            float32_t* max_feature_value_out,
        ) noexcept nogil
        inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil
        inline intp_t partition_samples(self, float64_t current_threshold) noexcept nogil        
        inline void partition_samples_final(
            self,
            intp_t best_pos,
            float64_t best_threshold,
            intp_t best_feature,
            intp_t best_n_missing,
        ) noexcept nogil

        InitNodeSplitFunction _init_node_split
        SortSamplesAndFeatureValuesFunction _sort_samples_and_feature_values
        FindMinMaxFunction _find_min_max
        NextPFunction _next_p
        PartitionSamplesFunction _partition_samples
        PartitionSamplesFinalFunction _partition_samples_final


cdef class DensePartitioner(Partitioner):
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef:
        const float32_t[:, :] X


cdef class SparsePartitioner(Partitioner):
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef:
        const float32_t[::1] X_data
        const int32_t[::1] X_indices
        const int32_t[::1] X_indptr

        intp_t n_total_samples

        intp_t[::1] index_to_samples
        intp_t[::1] sorted_samples

        intp_t start_positive
        intp_t end_negative
        bint is_samples_sorted

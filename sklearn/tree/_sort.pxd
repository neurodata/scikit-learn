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

from ..utils._typedefs cimport float32_t, float64_t, intp_t, int8_t, int32_t, uint32_t

# Since we broke Partitioner out into its own module in order to reuse it, and since
# both Splitter and Partitioner use these sort functions, we break them out into
# their own files in order to avoid cyclic file dependency.

# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Sort n-element arrays pointed to by feature_values and samples, simultaneously,
# by the values in feature_values. Algorithm: Introsort (Musser, SP&E, 1997).
cdef void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil

cdef void swap(float32_t* feature_values, intp_t* samples, intp_t i, intp_t j) noexcept nogil
cdef void sparse_swap(intp_t[::1] index_to_samples, intp_t[::1] samples,
    intp_t pos_1, intp_t pos_2) noexcept nogil

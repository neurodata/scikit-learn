from ..utils._typedefs cimport float32_t, float64_t, intp_t, int8_t, int32_t, uint32_t


# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Sort n-element arrays pointed to by feature_values and samples, simultaneously,
# by the values in feature_values. Algorithm: Introsort (Musser, SP&E, 1997).
cdef void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil

cdef void swap(float32_t* feature_values, intp_t* samples, intp_t i, intp_t j) noexcept nogil
cdef void sparse_swap(intp_t[::1] index_to_samples, intp_t[::1] samples,
    intp_t pos_1, intp_t pos_2) noexcept nogil

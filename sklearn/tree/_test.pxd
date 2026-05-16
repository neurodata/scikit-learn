from libcpp.vector cimport vector

from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t

from ._tree cimport Node
from ._honesty cimport Interval as Cinterval


cdef class TestNode():
    cdef:
        public list bounds
        public int start_idx
        public int n


cdef class HonestyTester():
    cdef:
        Node* nodes
        vector[Cinterval] intervals
        const float32_t[:, :] X
        const intp_t[::1] samples

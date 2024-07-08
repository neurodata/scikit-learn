# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause

# See _events.pyx for details.

from libcpp.vector cimport vector
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t

ctypedef int EventType
ctypedef void* EventHandlerEnv
ctypedef void* EventData
ctypedef bint (*EventHandlerFunction)(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil

cdef struct EventHandlerClosure:
    EventHandlerFunction f
    EventHandlerEnv e

cdef class EventHandler:
    cdef int[:] event_types
    cdef EventHandlerClosure c

cdef class EventBroker:
    cdef vector[vector[EventHandlerClosure]] listeners
    cdef bint fire_event(self, EventType event_type, EventData event_data) noexcept nogil

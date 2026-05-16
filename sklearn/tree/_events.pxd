# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause

# See _events.pyx for details.

from libcpp.vector cimport vector
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t


# a simple, general purpose event broker.
#
# it utilizes a somewhat clunky interface built around an event handler closure
# struct, as we are trying to balance generality with execution speed, and in
# practice nothing's faster than simply applying a function pointer.
#
# the idea is we would like something like a closure for event handlers, so that
# we may bind instances to instance-specific parameter values, like say you have
# a "threshold" parameter and you would like threshold-dependent handler behavior,
# but you want this threshold configurable at runtime. so we keep this threshold
# parameter in an environment bound to a "closure" instance, which is just a struct
# with a pointer to the environment instance and handler function. now vectors of
# these closures are compact, fast to iterate through, and low overhead to execute.
#
# the idea with EventType is that you have an event broker handling a class of
# conceptually related events, like suppose "server" events, and EventType would
# typically be values from an enum like say:
#
# cdef enum ServerEvent:
#     SERVER_UP = 1
#     SERVER_DOWN = 2
#     SERVER_ON_FIRE = 3
#
# an assumption of the current implementation is that these enum values are small
# integers, and we use them to allocate and index into a listener vector.
#
# EventData is simply a pointer to whatever event payload information is relevant
# to your handler, and it is expected that event_type maps to an associated handler
# which knows what specific "concrete" type to cast its event_data to.

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
    cdef public int[:] event_types
    cdef EventHandlerClosure c

cdef class NullHandler(EventHandler):
    pass

cdef class EventBroker:
    cdef vector[vector[EventHandlerClosure]] listeners # listeners acts as a map from EventType to corresponding event handlers
    cdef bint fire_event(self, EventType event_type, EventData event_data) noexcept nogil

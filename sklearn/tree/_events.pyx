
# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause


cdef class EventBroker:
    def __cinit__(self, EventHandler[:] listeners, int[:] event_types):
        cdef int i, ct
        cdef list l

        self.listeners.resize(len(event_types) + 1)
        if(listeners is not None):
            for e in event_types:
                l = [j for j, _l in enumerate(listeners) if e in _l.events]
                ct = len(l)
                self.listeners[e].resize(ct)
                for i in range(ct):
                    self.listeners[e][i] = listeners[l[i]].c
        else:
            for e in event_types:
                self.listeners[e].resize(0)

    cdef bint fire_event(self, EventType event_type, EventData event_data) noexcept nogil:
        cdef bint result = True

        for l in self.listeners[event_type]:
            result = result and l.f(event_type, l.e, event_data)
        
        return result

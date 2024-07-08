# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause

# See _honesty.pyx for details.

from ._events cimport EventData, EventHandler, EventHandlerEnv, EventType
from ._splitter cimport Partitioner, Splitter
from ._splitter cimport NodeSplitEvent, NodeSortFeatureEventData, NodeSplitEventData
from ._splitter cimport SplitConditionEnv, SplitConditionFunction, SplitConditionClosure, SplitCondition
from ._tree cimport TreeBuildEvent, TreeBuildSetActiveParentEventData, TreeBuildAddNodeEventData

from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t

from libcpp.vector cimport vector


cdef struct Interval:
    intp_t start_idx
    intp_t n
    intp_t feature
    intp_t split_idx      # start of right child
    float64_t split_value

cdef class Views:
    cdef:
        const float32_t[:, :] X
        intp_t[::1] samples
        float32_t[::1] feature_values
        Partitioner partitioner

cdef struct HonestEnv:
    void* data_views
    vector[Interval] tree
    Interval* active_parent
    Interval active_node
    intp_t active_is_left

cdef class Honesty:
    cdef:
        object splitter_event_handlers # python list of EventHandler
        object split_conditions        # python list of SplitCondition
        object tree_event_handlers     # python list of EventHandler

        Views views
        HonestEnv env

cdef struct MinSamplesLeafConditionEnv:
    intp_t min_samples
    HonestEnv* honest_env


cdef class NodeSortFeatureHandler(EventHandler):
    pass

cdef class AddNodeHandler(EventHandler):
    pass

cdef class SetActiveParentHandler(EventHandler):
    pass

cdef class HonestMinSamplesLeafCondition(SplitCondition):
    cdef MinSamplesLeafConditionEnv _env

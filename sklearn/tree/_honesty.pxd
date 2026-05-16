# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause

# See _honesty.pyx for details.

# Here we cash in the architectural changes/additions we made to Splitter and
# TreeBuilder. We implement this as an honest module not dependent on any particular
# type of Tree so that it can be composed into any type of Tree.
#
# The general ideas are that we:
# 1. inject honest split rejection criteria into Splitter
# 2. listen to tree build events fired by TreeBuilder to build a shadow tree
#    which contains the honest sample
#
# So we implement honest split rejection criteria for injection into Splitter,
# and event handlers which construct the shadow tree in response to events fired
# by TreeBuilder.

from ._events cimport EventData, EventHandler, EventHandlerEnv, EventType
from ._partitioner cimport Partitioner
from ._splitter cimport (
    NodeSplitEvent,
    NodeSortFeatureEventData,
    NodeSplitEventData,
    Splitter,
    SplitConditionEnv,
    SplitConditionFunction,
    SplitConditionClosure,
    SplitCondition
)
from ._tree cimport (
    Tree,
    TreeBuildEvent,
    TreeBuildSetActiveParentEventData,
    TreeBuildAddNodeEventData
)

from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t

from libcpp.vector cimport vector


# We do a much simplified tree model, barely more than enough to define the
# partition extents in the honest-masked data array corresponding to the node's
# elements. We store it in a vector indexed by the corresponding node IDs in the
# "structure" tree.
cdef struct Interval:
    intp_t start_idx      # index into samples
    intp_t n
    intp_t feature
    intp_t split_idx      # start of right child
    float64_t split_value

cdef class Views:
    cdef:
        const float32_t[:, :] X
        const float32_t[:, ::1] y
        intp_t[::1] samples
        float32_t[::1] feature_values   # temp. array holding feature values
        Partitioner partitioner

cdef struct HonestEnv:
    void* data_views
    vector[Interval] tree
    intp_t node_count
    Interval* active_parent
    Interval active_node
    intp_t active_is_left

cdef class Honesty:
    cdef:
        public list splitter_event_handlers # python list of EventHandler
        public list presplit_conditions     # python list of SplitCondition
        public list postsplit_conditions    # python list of SplitCondition
        public list tree_event_handlers     # python list of EventHandler

        public Views views
        HonestEnv env


cdef class HonestTree(Tree):
    cdef public Tree target_tree


cdef struct TrivialEnv:
    vector[int32_t] event_types

cdef class TrivialHandler(EventHandler):
    cdef TrivialEnv _env

cdef class NodeSortFeatureHandler(EventHandler):
    pass

cdef class AddNodeHandler(EventHandler):
    pass

cdef class SetActiveParentHandler(EventHandler):
    pass

cdef class TrivialCondition(SplitCondition):
    pass


cdef struct MinSamplesLeafConditionEnv:
    intp_t min_samples
    HonestEnv* honest_env

cdef class HonestMinSamplesLeafCondition(SplitCondition):
    cdef MinSamplesLeafConditionEnv _env

from cython cimport cast
from libc.stdint cimport uintptr_t
from libc.math cimport floor, fmax, log2, pow, isnan, NAN

from ._criterion cimport BaseCriterion, Criterion
from ._partitioner cimport DensePartitioner, SparsePartitioner

cimport numpy as cnp
import numpy as np
from scipy.sparse import issparse


cdef class HonestTree(Tree):
    """args[0] must be target_tree of type Tree"""
    def __init__(self, intp_t n_features, cnp.ndarray n_classes, intp_t n_outputs, Tree target_tree, *args):
        self.target_tree = target_tree

    cpdef cnp.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        return self.target_tree.apply(X)


cdef class Honesty:
    def __cinit__(
        self,
        object X,
        object samples,
        intp_t min_samples_leaf,
        const unsigned char[::1] missing_values_in_feature_mask = None,
        Partitioner honest_partitioner = None,
        splitter_event_handlers : [EventHandler] = None,
        presplit_conditions : [SplitCondition] = None,
        postsplit_conditions : [SplitCondition] = None,
        tree_event_handlers : [EventHandler] = None
    ):
        if splitter_event_handlers is None:
            splitter_event_handlers = []
        if presplit_conditions is None:
            presplit_conditions = []
        if postsplit_conditions is None:
            postsplit_conditions = []
        if tree_event_handlers is None:
            tree_event_handlers = []

        self.env.node_count = 0
        self.views = Views()
        self.views.X = X
        self.views.samples = samples
        self.views.feature_values = np.empty(len(samples), dtype=np.float32)
        self.views.partitioner = (
            honest_partitioner if honest_partitioner is not None
            else Honesty.create_partitioner(
                X,
                samples,
                self.views.feature_values,
                missing_values_in_feature_mask
            )
        )
        self.env.data_views = <void*>self.views

        self.splitter_event_handlers = [NodeSortFeatureHandler(self)] + (
            splitter_event_handlers if splitter_event_handlers is not None else []
        )
        self.presplit_conditions = [HonestMinSamplesLeafCondition(self, min_samples_leaf)] + (
            presplit_conditions if presplit_conditions is not None else []
        )
        self.postsplit_conditions = [] + (
            postsplit_conditions if postsplit_conditions is not None else []
        )
        self.tree_event_handlers = [
            SetActiveParentHandler(self),
            AddNodeHandler(self)
        ] + (tree_event_handlers if tree_event_handlers is not None else [])


    @staticmethod
    def create_partitioner(X, samples, feature_values, missing_values_in_feature_mask):
        return SparsePartitioner(
            X, samples, feature_values, missing_values_in_feature_mask
        ) if issparse(X) else DensePartitioner(
            X, samples, feature_values, missing_values_in_feature_mask
        )
    
    # The Criterion classes are quite stateful, and since we wish to reuse them
    # to maintain behavior consistent with them, we have to do some implementational
    # shenanigans like this.
    def init_criterion(
        self,
        Criterion criterion,
        y,
        sample_weights,
        weighted_n_samples,
        sample_indices
    ):
        criterion.init(y, sample_weights, weighted_n_samples, sample_indices)

    def set_sample_pointers(self, Criterion criterion, intp_t start, intp_t end):
        criterion.set_sample_pointers(start, end)
    
    def init_sum_missing(self, Criterion criterion):
        criterion.init_sum_missing()
    
    def node_value(self, Tree tree, Criterion criterion, intp_t i):
        criterion.node_value(<float64_t*>(tree.value + i * tree.value_stride))
    
    def node_samples(self, Tree tree, Criterion criterion, intp_t i):
        criterion.node_samples(tree.value_samples[i])

    def get_node_count(self):
        return self.env.node_count

    def resize_tree(self, Tree tree, intp_t capacity):
        tree._resize(capacity)
    
    def get_node_range(self, i):
        return (
            self.env.tree[i].start_idx,
            self.env.tree[i].start_idx + self.env.tree[i].n
        )
    
    def is_leaf(self, i):
        return self.env.tree[i].feature == -1
    
    @staticmethod
    def get_value_samples_ndarray(Tree tree, intp_t node_id):
        return tree._get_value_samples_ndarray(node_id)


cdef bint _handle_trivial(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    cdef bint result = False
    cdef TrivialEnv* env = <TrivialEnv*>handler_env

    with gil:
        print("in _handle_trivial")
    
    for i in range(env.event_types.size()):
        result = result | env.event_types[i]
    
    return result


cdef class TrivialHandler(EventHandler):
    def __cinit__(self, event_types : [EventType]):
        self.event_types = np.array(event_types, dtype=np.int32)

        self._env.event_types.resize(len(event_types))
        for i in range(len(event_types)):
            self._env.event_types[i] = event_types[i]
        
        self.c.f = _handle_trivial
        self.c.e = &self._env


cdef bint _handle_set_active_parent(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != TreeBuildEvent.SET_ACTIVE_PARENT:
        return True
    
    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef TreeBuildSetActiveParentEventData* data = <TreeBuildSetActiveParentEventData*>event_data
    cdef Interval* node = &env.active_node

    if (<int32_t>data.parent_node_id) >= (<int32_t>env.tree.size()):
        return False

    env.active_is_left = data.child_is_left

    node.feature = -1
    node.split_idx = 0
    node.split_value = NAN

    if data.parent_node_id < 0:
        env.active_parent = NULL
        node.start_idx = 0
        node.n = (<Views>env.data_views).samples.shape[0]
    else:
        env.active_parent = &(env.tree[data.parent_node_id])
        if env.active_is_left:
            node.start_idx = env.active_parent.start_idx
            node.n = env.active_parent.split_idx - env.active_parent.start_idx
        else:
            node.start_idx = env.active_parent.split_idx
            node.n = env.active_parent.n - env.active_parent.split_idx

    (<Views>env.data_views).partitioner.init_node_split(node.start_idx, node.start_idx + node.n)

    return True

cdef class SetActiveParentHandler(EventHandler):
    def __cinit__(self, Honesty h):
        self.event_types = np.array([TreeBuildEvent.SET_ACTIVE_PARENT], dtype=np.int32)

        self.c.f = _handle_set_active_parent
        self.c.e = &h.env


cdef bint _handle_sort_feature(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != NodeSplitEvent.SORT_FEATURE:
        return True
    
    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef NodeSortFeatureEventData* data = <NodeSortFeatureEventData*>event_data
    cdef Interval* node = &env.active_node

    node.feature = data.feature
    node.split_idx = 0
    node.split_value = NAN

    (<Views>env.data_views).partitioner.sort_samples_and_feature_values(node.feature)

    return True

# When the structure tree sorts by a feature, we must do the same
cdef class NodeSortFeatureHandler(EventHandler):
    def __cinit__(self, Honesty h):
        self.event_types = np.array([NodeSplitEvent.SORT_FEATURE], dtype=np.int32)

        self.c.f = _handle_sort_feature
        self.c.e = &h.env


cdef bint _handle_add_node(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != TreeBuildEvent.ADD_NODE:
        return True

    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef const float32_t[:, :] X = (<Views>env.data_views).X
    cdef intp_t[::1] samples = (<Views>env.data_views).samples
    cdef float64_t h, feature_value
    cdef intp_t i, n_left, n_missing, size = env.tree.size()
    cdef TreeBuildAddNodeEventData* data = <TreeBuildAddNodeEventData*>event_data
    cdef Interval *interval = NULL
    cdef Interval *parent = NULL

    if data.node_id >= size:
        # as a heuristic, assume a complete tree and add a level
        h = floor(fmax(0, log2(size)))
        env.tree.resize(size + <intp_t>pow(2, h + 1))

    interval = &(env.tree[data.node_id])
    interval.feature = data.feature
    interval.split_value = data.split_point

    if data.parent_node_id < 0:
        # the node being added is the tree root
        interval.start_idx = 0
        interval.n = samples.shape[0]
    else:
        parent = &(env.tree[data.parent_node_id])

        if data.is_left:
            interval.start_idx = parent.start_idx
            interval.n = parent.split_idx - parent.start_idx
        else:
            interval.start_idx = parent.split_idx
            interval.n = parent.n - (parent.split_idx - parent.start_idx)

    # We also reuse Partitioner. *We* don't need to sort to find the split pos we'll
    # need for partitioning, but the partitioner internals are so stateful we had
    # better just do it to ensure that it's in the expected state
    (<Views>env.data_views).partitioner.init_node_split(interval.start_idx, interval.start_idx + interval.n)
    (<Views>env.data_views).partitioner.sort_samples_and_feature_values(interval.feature)

    # count n_left to find split pos
    n_left = 0
    i = interval.start_idx
    feature_value = X[samples[i], interval.feature]

    while (not isnan(feature_value)) and feature_value < interval.split_value and i < interval.start_idx + interval.n:
        n_left += 1
        i += 1
        feature_value = X[samples[i], interval.feature]

    interval.split_idx = interval.start_idx + n_left

    (<Views>env.data_views).partitioner.partition_samples_final(
        interval.split_idx, interval.split_value, interval.feature, (<Views>env.data_views).partitioner.n_missing
        )
    
    env.node_count += 1


cdef class AddNodeHandler(EventHandler):
    def __cinit__(self, Honesty h):
        self.event_types = np.array([TreeBuildEvent.ADD_NODE], dtype=np.int32)

        self.c.f = _handle_add_node
        self.c.e = &h.env


cdef bint _trivial_condition(
    Splitter splitter,
    intp_t split_feature,
    intp_t split_pos,
    float64_t split_value,
    intp_t n_missing,
    bint missing_go_to_left,
    float64_t lower_bound,
    float64_t upper_bound,
    SplitConditionEnv split_condition_env
) noexcept nogil:
    return True

cdef class TrivialCondition(SplitCondition):
    def __cinit__(self):
        self.c.f = _trivial_condition
        self.c.e = NULL


cdef bint _honest_min_sample_leaf_condition(
    Splitter splitter,
    intp_t split_feature,
    intp_t split_pos,
    float64_t split_value,
    intp_t n_missing,
    bint missing_go_to_left,
    float64_t lower_bound,
    float64_t upper_bound,
    SplitConditionEnv split_condition_env
) noexcept nogil:
    cdef MinSamplesLeafConditionEnv* env = <MinSamplesLeafConditionEnv*>split_condition_env
    cdef Interval* node = &env.honest_env.active_node

    cdef intp_t min_samples_leaf = env.min_samples
    cdef intp_t end_non_missing, n_left, n_right

    # we don't care about n_missing in the structure set
    n_missing = (<Views>env.honest_env.data_views).partitioner.n_missing
    end_non_missing = node.start_idx + node.n - n_missing

    # we don't care about split_pos in the structure set,
    # need to scan forward in the honest set based on split_value to find it
    while node.split_idx < node.start_idx + node.n and (<Views>env.honest_env.data_views).X[(<Views>env.honest_env.data_views).samples[node.split_idx], node.feature] <= split_value:
        node.split_idx += 1
    
    if missing_go_to_left:
        n_left = node.split_idx - node.start_idx + n_missing
        n_right = end_non_missing - node.split_idx
    else:
        n_left = node.split_idx - node.start_idx
        n_right = end_non_missing - node.split_idx + n_missing

    # Reject if min_samples_leaf is not guaranteed
    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        #with gil:
        #    print("returning False")
        return False

    return True

# Check that the honest set will have sufficient samples on each side of this
# candidate split.
cdef class HonestMinSamplesLeafCondition(SplitCondition):
    def __cinit__(self, Honesty h, intp_t min_samples):
        self._env.min_samples = min_samples
        self._env.honest_env = &h.env

        self.c.f = _honest_min_sample_leaf_condition
        self.c.e = &self._env

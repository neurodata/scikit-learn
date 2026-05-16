# Authors: Haoyin Xu <haoyinxu@gmail.com>
#          Samuel Carliles <scarlil1@jhu.edu>
#
# Adopted from: https://github.com/neurodata/honest-forests

# An honest classification tree implemented by inheriting BaseDecisionTree and
# including the honesty module. The general idea is that:
#
# 1. The interface looks mostly like a regular DecisionTree, and we inherit as
#    much of the implementation as we can.
# 2. Rather than actually being our own tree however, we have a target tree for
#    learning the structure which is just a regular DecisionTree trained on the
#    structure sample, and an honesty instance which grows the shadow tree described
#    in the honesty module.

import numpy as np
from numpy import float32 as DTYPE

from ..base import _fit_context, is_classifier
from ..model_selection import StratifiedShuffleSplit
from ..utils import compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets

from ._classes import (
    BaseDecisionTree,
    CRITERIA_CLF, CRITERIA_REG, DENSE_SPLITTERS, SPARSE_SPLITTERS
)
from ._honesty import HonestTree, Honesty
from ._tree import DOUBLE, Tree

import inspect


# note: max_n_classes is the maximum number of classes observed
# in any response variable dimension
class HonestDecisionTree(BaseDecisionTree):
    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "target_tree_class": "no_validation",
        "target_tree_kwargs": [dict],
        "honest_fraction": [Interval(RealNotInt, 0.0, 1.0, closed="both")],
        "honest_prior": [StrOptions({"empirical", "uniform", "ignore"})],
        "stratify": ["boolean"],
    }

    def __init__(
        self,
        *,
        criterion=None,
        target_tree_class=None,
        target_tree_kwargs=None,
        random_state=None,
        honest_fraction=0.5,
        honest_prior="empirical",
        stratify=False
    ):
        self.criterion = criterion
        self.target_tree_class = target_tree_class
        self.target_tree_kwargs = target_tree_kwargs if target_tree_kwargs is not None else {}

        self.random_state = random_state
        self.honest_fraction = honest_fraction
        self.honest_prior = honest_prior
        self.stratify = stratify

        # TODO: unwind this whole gross antipattern
        if target_tree_class is not None:
            HonestDecisionTree._target_tree_hack(self, target_tree_class, **target_tree_kwargs)
    
    # In order to inherit behavior from BaseDecisionTree, we must satisfy a lot of
    # pythonic introspective attribute assumptions. This was the lowest effort way
    # that came to mind.
    @staticmethod
    def _target_tree_hack(honest_tree, target_tree_class, **kwargs):
        honest_tree.target_tree_class = target_tree_class
        honest_tree.target_tree = target_tree_class(**kwargs)

        # copy over the attributes of the target tree
        for attr_name in vars(honest_tree.target_tree):
            setattr(
                honest_tree,
                attr_name,
                getattr(honest_tree.target_tree, attr_name, None)
            )

        if is_classifier(honest_tree.target_tree):
            honest_tree._estimator_type = honest_tree.target_tree._estimator_type
            honest_tree.predict_proba = honest_tree.target_tree.predict_proba
            honest_tree.predict_log_proba = honest_tree.target_tree.predict_log_proba

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
        classes=None
    ):
        return self.fit(
            X, y, sample_weight, check_input, missing_values_in_feature_mask, classes
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
        classes=None,
    ):
        """Build an honest tree from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

        Returns
        -------
        self : HonestTree
            Fitted tree estimator.
        """

        # run this again because of the way ensemble creates estimators
        HonestDecisionTree._target_tree_hack(self, self.target_tree_class, **self.target_tree_kwargs)
        target_bta = self.target_tree._prep_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_input=check_input,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
            classes=classes
        )

        # TODO: go fix TODO in classes.py line 636
        if target_bta.n_classes is None:
            target_bta.n_classes = np.array(
                [1] * self.target_tree.n_outputs_,
                dtype=np.intp
            )

        # Determine output settings
        self._init_output_shape(target_bta.X, target_bta.y, target_bta.classes)

        # obtain the structure sample weights
        sample_weights_structure, sample_weights_honest = self._partition_honest_indices(
            target_bta.y,
            target_bta.sample_weight
        )

        # create honesty, set up listeners in target tree
        self.honesty = Honesty(
            target_bta.X,
            self.honest_indices_,
            target_bta.min_samples_leaf,
            missing_values_in_feature_mask = target_bta.missing_values_in_feature_mask
        )

        self.target_tree.presplit_conditions = self.honesty.presplit_conditions
        self.target_tree.postsplit_conditions = self.honesty.postsplit_conditions
        self.target_tree.splitter_listeners = self.honesty.splitter_event_handlers
        self.target_tree.tree_build_listeners = self.honesty.tree_event_handlers

        # Learn structure on subsample
        # XXX: this allows us to use BaseDecisionTree without partial_fit API
        try:
            self.target_tree.fit(
                target_bta.X,
                target_bta.y,
                sample_weight=sample_weights_structure,
                check_input=check_input,
                classes=target_bta.classes
            )
        except Exception:
            self.target_tree.fit(
                target_bta.X,
                target_bta.y,
                sample_weight=sample_weights_structure,
                check_input=check_input
            )

        # more pythonic introspection minutiae
        setattr(
            self,
            "classes_",
            getattr(self.target_tree, "classes_", None)
        )

        n_samples = target_bta.X.shape[0]
        samples = np.empty(n_samples, dtype=np.intp)
        weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weights_honest[i] != 0.0:
                samples[j] = i
                j += 1

            weighted_n_samples += sample_weights_honest[i]

        # more pythonic introspection minutiae
        # fingers crossed sklearn.utils.validation.check_is_fitted doesn't
        # change its behavior
        self.tree_ = HonestTree(
            self.target_tree.n_features_in_,
            target_bta.n_classes,
            self.target_tree.n_outputs_,
            self.target_tree.tree_
        )
        self.honesty.resize_tree(self.tree_, self.honesty.get_node_count())
        self.tree_.node_count = self.honesty.get_node_count()

        # Criterion is very stateful, so do all the instantiation and initialization
        criterion = BaseDecisionTree._create_criterion(
            self.target_tree,
            n_outputs=target_bta.y.shape[1],
            n_samples=target_bta.X.shape[0],
            n_classes=target_bta.n_classes
        )
        self.honesty.init_criterion(
            criterion,
            target_bta.y,
            sample_weights_honest,
            weighted_n_samples,
            self.honest_indices_
        )

        for i in range(self.honesty.get_node_count()):
            start, end = self.honesty.get_node_range(i)
            self.honesty.set_sample_pointers(criterion, start, end)

            if missing_values_in_feature_mask is not None:
                self.honesty.init_sum_missing(criterion)
            
            self.honesty.node_value(self.tree_, criterion, i)

            if self.honesty.is_leaf(i):
                self.honesty.node_samples(self.tree_, criterion, i)

        # more pythonic introspection minutiae
        setattr(
            self,
            "__sklearn_is_fitted__",
            lambda: True
        )
 
        return self

    
    def _init_output_shape(self, X, y, classes=None):
        # Determine output settings
        self.n_samples_, self.n_features_in_ = X.shape

        # Do preprocessing if 'y' is passed
        is_classification = False
        if y is not None:
            is_classification = is_classifier(self)
            y = np.atleast_1d(y)
            expanded_class_weight = None

            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))

            self.n_outputs_ = y.shape[1]

            if is_classification:
                check_classification_targets(y)
                y = np.copy(y)

                self.classes_ = []
                self.n_classes_ = []

                if self.class_weight is not None:
                    y_original = np.copy(y)

                y_encoded = np.zeros(y.shape, dtype=int)
                if classes is not None:
                    classes = np.atleast_1d(classes)
                    if classes.ndim == 1:
                        classes = np.array([classes])

                    for k in classes:
                        self.classes_.append(np.array(k))
                        self.n_classes_.append(np.array(k).shape[0])

                    for i in range(self.n_samples_):
                        for j in range(self.n_outputs_):
                            y_encoded[i, j] = np.where(self.classes_[j] == y[i, j])[0][
                                0
                            ]
                else:
                    for k in range(self.n_outputs_):
                        classes_k, y_encoded[:, k] = np.unique(
                            y[:, k], return_inverse=True
                        )
                        self.classes_.append(classes_k)
                        self.n_classes_.append(classes_k.shape[0])

                y = y_encoded

                if self.class_weight is not None:
                    expanded_class_weight = compute_sample_weight(
                        self.class_weight, y_original
                    )

                self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
                self._n_classes_ = self.n_classes_
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)

            if len(y) != self.n_samples_:
                raise ValueError(
                    "Number of labels=%d does not match number of samples=%d"
                    % (len(y), self.n_samples_)
                )


    def _partition_honest_indices(self, y, sample_weight):
        # Account for bootstrapping too
        if sample_weight is None:
            structure_weight = np.ones((len(y),), dtype=np.float64)
            honest_weight = np.ones((len(y),), dtype=np.float64)
        else:
            structure_weight = np.array(sample_weight)
            honest_weight = np.array(sample_weight)

        nonzero_indices = np.where(structure_weight > 0)[0]

        # sample the structure indices
        if self.stratify:
            ss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.honest_fraction, random_state=self.random_state
            )
            for structure_idx, _ in ss.split(
                np.zeros((len(nonzero_indices), 1)), y[nonzero_indices]
            ):
                self.structure_indices_ = nonzero_indices[structure_idx]

        else:
            rng = np.random.default_rng(self.random_state)
            self.structure_indices_ = rng.choice(
                nonzero_indices,
                int((1 - self.honest_fraction) * len(nonzero_indices)),
                replace=False,
            )

        honest_weight[self.structure_indices_] = 0

        self.honest_indices_ = np.setdiff1d(nonzero_indices, self.structure_indices_)
        structure_weight[self.honest_indices_] = 0

        return structure_weight, honest_weight

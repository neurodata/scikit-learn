"""Histogram-based Random Forest for classification and regression"""
import numbers#
from warnings import catch_warnings, simplefilter, warn#
import threading#
from abc import ABC, ABCMeta, abstractmethod#
from functools import partial#
import warnings#

import numpy as np#
from scipy.sparse import issparse #
from scipy.sparse import hstack as sparse_hstack #
from joblib import Parallel#
from timeit import default_timer as time #

from ...base import (BaseEstimator, 
                    RegressorMixin, 
                    ClassifierMixin, 
                    is_classifier)
from ...utils.validation import (check_is_fitted,
                                check_consistent_length,
                                _check_sample_weight,
                                _deprecate_positional_args)
from ...utils.multiclass import check_classification_targets
from ...metrics import check_scoring, accuracy_score, r2_score
from ...model_selection import train_test_split #
from ...preprocessing import LabelEncoder, OneHotEncoder
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder

from ..tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                    ExtraTreeClassifier, ExtraTreeRegressor)
from ._gradient_boosting import _update_raw_predictions
from ..tree._tree import DTYPE, DOUBLE
from ..utils import check_random_state, compute_sample_weight, deprecated, resample
from ..exceptions import DataConversionWarning
from ._base import BaseEnsemble, _partition_estimators
from ..utils.fixes import delayed
from ..utils.fixes import _joblib_parallel_args
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils.validation import _deprecate_positional_args
from .common import Y_DTYPE, X_DTYPE, X_BINNED_DTYPE

from .binning import _BinMapper#
from .grower import TreeGrower#
from .loss import _LOSSES#
from .loss import BaseLoss#


class HistRandomForest(BaseEstimator, ABC):
    @abstractmethod
    def __init__(self,
                base_estimator,
                n_estimators=100, *,
                estimator_params=tuple(),
                bootstrap=False,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=None,
                max_samples=None,
                max_bins=255):
        self.max_bins = max_bins 
    
    def _validate_parameters(self):
    
    def apply(self, X):
    
    def decision_path(self, X):
        
    def _get_oob_predictions(tree, X):
        
    def _set_oob_score_and_attributes(self, X, y):
        
    def fit(self, X, y, sample_weight=None):
        n_bins = self.max_bins + 1  # + 1 for missing values
        self._bin_mapper = _BinMapper(
            n_bins=n_bins,
            is_categorical=self.is_categorical_,
            known_categories=known_categories,
            random_state=self._random_seed)
        X_binned_train = self._bin_data(X_train, is_training_data=True)
        if X_val is not None:
            X_binned_val = self._bin_data(X_val, is_training_data=False)
        else:
            X_binned_val = None
        
    def predict(self, X):
        
    def predict_proba(self, X):
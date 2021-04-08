import numpy as np

class TreeSplitInfo():
    def __init__(self, gain, feature_idx, bin_idx, 
                missing_go_to_left, n_samples_left, 
                n_samples_right, value_left, value_right,
                is_categorical):
        self.gain = gain
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx
        self.missing_go_to_left = missing_go_to_left
        self.n_samples_left = n_samples_left
        self.n_samples_right = n_samples_right
        self.value_left = value_left
        self.value_right = value_right
        self.is_categorical = is_categorical
        self.left_cat_bitset = left_cat_bitset
    
class TreeSplitter:
    def __init__(self, X_binned, 
                n_bins_non_missing, 
                missing_values_bin_idx, 
                has_missing_values, 
                is_categorical, 
                monotonic_cst, 
                l2_regularization,
                min_samples_leaf, 
                min_gain_to_split):
        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        self.n_bins_non_missing = n_bins_non_missing
        self.missing_values_bin_idx = missing_values_bin_idx
        self.has_missing_values = has_missing_values
        self.monotonic_cst = monotonic_cst
        self.is_categorical = is_categorical
        self.l2_regularization = l2_regularization
        self.min_hessian_to_split = min_hessian_to_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.hessians_are_constant = hessians_are_constant
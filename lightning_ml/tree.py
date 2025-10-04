"""
Decision Tree Implementation using PyTorch for GPU acceleration
Supports both Classification and Regression
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass
from .base_model import BaseTreeModel


@dataclass
class TreeNode:
    """Node in the decision tree."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: Optional[Union[float, int]] = None  # Leaf node value
    impurity: float = 0.0
    n_samples: int = 0
    # For better probability estimates in classification
    class_counts: Optional[dict] = None
    
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeBase(BaseTreeModel):
    """
    Base class for Decision Trees with GPU acceleration.
    """
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = None,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize decision tree.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features to consider ('sqrt', 'log2', int, None)
            random_state: Random seed for reproducibility
            device: Computation device
        """
        super().__init__(max_depth, min_samples_split, min_samples_leaf, device)
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
    
    def _select_features(self, n_features: int) -> np.ndarray:
        """
        Select subset of features for splitting.
        
        Args:
            n_features: Total number of features
            
        Returns:
            Array of feature indices to consider
        """
        if self.max_features is None:
            return np.arange(n_features)
        elif self.max_features == 'sqrt':
            n = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            n = min(self.max_features, n_features)
        else:
            n = n_features
        
        return np.random.choice(n_features, n, replace=False)
    
    def _calculate_impurity(self, y: torch.Tensor) -> float:
        """Calculate impurity (to be overridden by child classes)."""
        raise NotImplementedError
    
    def _calculate_leaf_value(self, y: torch.Tensor) -> Union[float, int]:
        """Calculate leaf node value (to be overridden by child classes)."""
        raise NotImplementedError
    
    def _find_best_split(self, 
                        X: torch.Tensor, 
                        y: torch.Tensor,
                        feature_indices: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best split for current node.
        
        Args:
            X: Feature tensor
            y: Target tensor
            feature_indices: Indices of features to consider
            
        Returns:
            Tuple of (best_feature_idx, best_threshold, best_impurity_gain)
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        n_samples = len(y)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            
            # Get unique values for potential thresholds
            unique_values = torch.unique(feature_values, sorted=True)
            
            if len(unique_values) <= 1:
                continue
            
            # Use midpoints between consecutive unique values as thresholds
            thresholds = []
            if len(unique_values) > 10:
                # Sample thresholds for efficiency
                indices = torch.linspace(0, len(unique_values) - 1, 10).long()
                sampled_values = unique_values[indices]
                thresholds = sampled_values.cpu().numpy()
            else:
                # Use all midpoints
                for i in range(len(unique_values) - 1):
                    threshold = (unique_values[i] + unique_values[i + 1]) / 2
                    thresholds.append(threshold.item())
            
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = left_mask.sum().item()
                n_right = right_mask.sum().item()
                
                # Check minimum samples
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(threshold)
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, 
                   X: torch.Tensor, 
                   y: torch.Tensor, 
                   depth: int = 0) -> TreeNode:
        """
        Recursively build decision tree.
        
        Args:
            X: Feature tensor
            y: Target tensor
            depth: Current depth
            
        Returns:
            TreeNode
        """
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Create node
        node = TreeNode(
            impurity=self._calculate_impurity(y),
            n_samples=n_samples
        )
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (node.impurity == 0.0):
            node.value = self._calculate_leaf_value(y)
            return node
        
        # Find best split
        feature_indices = self._select_features(n_features)
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, y, feature_indices
        )
        
        # If no good split found, create leaf
        if best_feature is None or best_gain <= 0:
            node.value = self._calculate_leaf_value(y)
            return node
        
        # Split data and build subtrees
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _traverse_tree(self, x: torch.Tensor, node: TreeNode) -> Union[float, int]:
        """
        Traverse tree for a single sample.
        
        Args:
            x: Single sample features
            node: Current node
            
        Returns:
            Prediction value
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def _calculate_feature_importances(self, node: Optional[TreeNode] = None, 
                                      total_samples: Optional[int] = None) -> np.ndarray:
        """Calculate feature importances recursively."""
        if node is None:
            node = self.tree_
            total_samples = node.n_samples
            self.feature_importances_ = np.zeros(self.n_features_in_)
        
        if node.is_leaf():
            return self.feature_importances_
        
        # Calculate impurity decrease
        if node.left and node.right:
            left_impurity = node.left.impurity * node.left.n_samples
            right_impurity = node.right.impurity * node.right.n_samples
            weighted_child_impurity = (left_impurity + right_impurity) / node.n_samples
            
            impurity_decrease = node.impurity - weighted_child_impurity
            importance = (node.n_samples / total_samples) * impurity_decrease
            
            self.feature_importances_[node.feature_idx] += importance
            
            # Recurse
            self._calculate_feature_importances(node.left, total_samples)
            self._calculate_feature_importances(node.right, total_samples)
        
        return self.feature_importances_
    
    def get_depth(self, node: Optional[TreeNode] = None) -> int:
        """Get maximum depth of the tree."""
        if node is None:
            node = self.tree_
        
        if node is None or node.is_leaf():
            return 0
        
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))
    
    def get_n_leaves(self, node: Optional[TreeNode] = None) -> int:
        """Get number of leaf nodes."""
        if node is None:
            node = self.tree_
        
        if node is None:
            return 0
        
        if node.is_leaf():
            return 1
        
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)
    
    def get_params(self):
        params = super().get_params()
        params.update({
            'max_features': self.max_features,
            'random_state': self.random_state
        })
        return params


class DecisionTreeRegressor(DecisionTreeBase):
    """
    Decision Tree Regressor with GPU support.
    
    Args:
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Number of features to consider
        criterion: Split criterion ('mse' or 'mae')
        random_state: Random seed
        device: Computation device
    """
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = None,
                 criterion: str = 'mse',
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, 
                        max_features, random_state, device)
        self.criterion = criterion
    
    def _calculate_impurity(self, y: torch.Tensor) -> float:
        """Calculate MSE or MAE."""
        if len(y) == 0:
            return 0.0
        
        if self.criterion == 'mse':
            mean = y.mean()
            return ((y - mean) ** 2).mean().item()
        else:  # mae
            median = y.median().values if hasattr(y.median(), 'values') else y.median()
            return (y - median).abs().mean().item()
    
    def _calculate_leaf_value(self, y: torch.Tensor) -> float:
        """Calculate mean for leaf node."""
        return y.mean().item()
    
    # ------------------ DecisionTreeRegressor ------------------

    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'DecisionTreeRegressor':
        """
        Fit decision tree regressor.
        """
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze()
        
        self._validate_input(X, y)
        
        # Update feature info first
        self._update_fit_info(X)
        
        # Build tree
        if verbose:
            print("Building decision tree...")
        
        self.tree_ = self._build_tree(X, y)
        
        # Initialize and calculate feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.feature_importances_ = self._calculate_feature_importances()
        
        # Normalize importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        if verbose:
            print(f"Decision Tree Regressor fitted:")
            print(f"  Features: {self.n_features_in_}")
            print(f"  Depth: {self.get_depth()}")
            print(f"  Leaves: {self.get_n_leaves()}")
        
        return self
        
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        predictions = np.array([
            self._traverse_tree(X[i], self.tree_) for i in range(len(X))
        ])
        
        return predictions
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Test features
            y: True targets
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        if y.ndim > 1:
            y = y.squeeze()
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class DecisionTreeClassifier(DecisionTreeBase):
    """
    Decision Tree Classifier with GPU support.
    
    Args:
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Number of features to consider
        criterion: Split criterion ('gini' or 'entropy')
        random_state: Random seed
        device: Computation device
    """
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = None,
                 criterion: str = 'gini',
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, 
                        max_features, random_state, device)
        self.criterion = criterion
        self.classes_ = None
        self.n_classes_ = None
    
    def _calculate_impurity(self, y: torch.Tensor) -> float:
        """Calculate Gini or Entropy."""
        if len(y) == 0:
            return 0.0
        
        _, counts = torch.unique(y, return_counts=True)
        probabilities = counts.float() / len(y)
        
        if self.criterion == 'gini':
            return (1 - (probabilities ** 2).sum()).item()
        else:  # entropy
            # Add small epsilon to avoid log(0)
            return -(probabilities * torch.log2(probabilities + 1e-10)).sum().item()
    
    def _calculate_leaf_value(self, y: torch.Tensor) -> int:
        """Calculate mode (most common class) for leaf node."""
        values, counts = torch.unique(y, return_counts=True)
        mode_idx = counts.argmax()
        return values[mode_idx].item()
    
    def _store_class_counts(self, y: torch.Tensor, node: TreeNode):
        """Store class distribution in leaf nodes for probability estimates."""
        if node.is_leaf():
            values, counts = torch.unique(y, return_counts=True)
            node.class_counts = {
                int(val.item()): int(count.item()) 
                for val, count in zip(values, counts)
            }
    
    def _build_tree_with_counts(self, X: torch.Tensor, y: torch.Tensor, 
                                depth: int = 0) -> TreeNode:
        """Build tree and store class counts in leaf nodes."""
        node = self._build_tree(X, y, depth)
        
        if node.is_leaf():
            values, counts = torch.unique(y, return_counts=True)
            node.class_counts = {
                int(val.item()): int(count.item()) 
                for val, count in zip(values, counts)
            }
        
        return node
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            verbose: bool = False) -> 'DecisionTreeClassifier':
        """
        Fit decision tree classifier.
        """
        X = self._to_tensor(X, dtype=torch.float32)
        y_np = y if isinstance(y, np.ndarray) else self._to_numpy(y)
        
        # Get unique classes
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        y = self._to_tensor(y_np, dtype=torch.long)
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze()
        
        self._validate_input(X, y)
        
        # Update feature info first
        self._update_fit_info(X)
        
        # Build tree with class counts
        if verbose:
            print("Building decision tree...")
        
        self.tree_ = self._build_tree_with_counts(X, y)
        
        # Initialize and calculate feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.feature_importances_ = self._calculate_feature_importances()
        
        # Normalize importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        if verbose:
            print(f"Decision Tree Classifier fitted:")
            print(f"  Classes: {self.n_classes_}")
            print(f"  Features: {self.n_features_in_}")
            print(f"  Depth: {self.get_depth()}")
            print(f"  Leaves: {self.get_n_leaves()}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        predictions = np.array([
            self._traverse_tree(X[i], self.tree_) for i in range(len(X))
        ])
        
        return predictions.astype(int)
    
    def _traverse_tree_proba(self, x: torch.Tensor, node: TreeNode) -> np.ndarray:
        """Traverse tree and return class probabilities."""
        if node.is_leaf():
            proba = np.zeros(self.n_classes_)
            if node.class_counts:
                total = sum(node.class_counts.values())
                for class_idx, count in node.class_counts.items():
                    # Find the position of this class in self.classes_
                    pos = np.where(self.classes_ == class_idx)[0]
                    if len(pos) > 0:
                        proba[pos[0]] = count / total
            else:
                # Fallback: one-hot encoding
                proba[node.value] = 1.0
            return proba
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree_proba(x, node.left)
        else:
            return self._traverse_tree_proba(x, node.right)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        probas = np.array([
            self._traverse_tree_proba(X[i], self.tree_) for i in range(len(X))
        ])
        
        return probas
    
    def score(self, X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        if y.ndim > 1:
            y = y.squeeze()
        
        return np.mean(predictions == y)
"""
Decision Tree Implementation using PyTorch for GPU acceleration
Supports both Classification and Regression
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass
from .base_model import BaseTreeModel
import logging

logger = logging.getLogger(__name__)


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
        """Calculate impurity (must be implemented by child classes)."""
        raise NotImplementedError("Child classes must implement _calculate_impurity")
    
    def _calculate_leaf_value(self, y: torch.Tensor) -> Union[float, int]:
        """Calculate leaf node value (must be implemented by child classes)."""
        raise NotImplementedError("Child classes must implement _calculate_leaf_value")
    
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
    
    def get_params(self, deep: bool = True):
        params = super().get_params(deep=deep)
        params.update({
            'max_features': self.max_features,
            'random_state': self.random_state
        })
        return params
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Note: Decision trees don't store tensors permanently like neural networks.
        This method updates the device attribute for future tensor operations.
        
        Args:
            device: Target device ('cuda', 'cpu', 'mps', or torch.device object)
            
        Returns:
            self for method chaining
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Convert feature_importances_ if it exists and is a tensor
        if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None:
            if isinstance(self.feature_importances_, torch.Tensor):
                self.feature_importances_ = self.feature_importances_.to(self.device)
        
        logger.info(f"Model device set to: {self.device}")
        return self
    
    def _tree_to_dict(self, node: Optional[TreeNode]) -> Optional[dict]:
        """
        Recursively convert tree structure to dictionary for serialization.
        
        Args:
            node: TreeNode to convert
            
        Returns:
            Dictionary representation of the tree
        """
        if node is None:
            return None
        
        return {
            'feature_idx': node.feature_idx,
            'threshold': node.threshold,
            'value': node.value,
            'impurity': node.impurity,
            'n_samples': node.n_samples,
            'class_counts': node.class_counts,
            'left': self._tree_to_dict(node.left),
            'right': self._tree_to_dict(node.right)
        }

    def _dict_to_tree(self, tree_dict: Optional[dict]) -> Optional[TreeNode]:
        """
        Recursively convert dictionary back to tree structure.
        
        Args:
            tree_dict: Dictionary representation of tree
            
        Returns:
            TreeNode object
        """
        if tree_dict is None:
            return None
        
        node = TreeNode(
            feature_idx=tree_dict['feature_idx'],
            threshold=tree_dict['threshold'],
            value=tree_dict['value'],
            impurity=tree_dict['impurity'],
            n_samples=tree_dict['n_samples'],
            class_counts=tree_dict['class_counts']
        )
        
        node.left = self._dict_to_tree(tree_dict['left'])
        node.right = self._dict_to_tree(tree_dict['right'])
        
        return node

    def save_model(self, filepath: str):
        """
        Save decision tree model to file.
        
        Args:
            filepath: Path where the model will be saved
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model")
            import warnings
            warnings.warn("Saving unfitted model - predictions may not work correctly")
        
        # Convert tree structure to dictionary
        tree_dict = self._tree_to_dict(self.tree_)
        
        # Build state dictionary
        state = {
            'tree': tree_dict,
            'params': self.get_params(),
            'training_history': self._training_history,
            'is_fitted': self.is_fitted,
            'model_class': self.__class__.__name__,
            'n_features_in_': self.n_features_in_,
            'n_samples_seen_': self.n_samples_seen_,
            'feature_importances_': self.feature_importances_
        }
        
        # Add classifier-specific attributes if they exist
        if hasattr(self, 'classes_'):
            state['classes_'] = self.classes_
            state['n_classes_'] = self.n_classes_
        
        # Add criterion
        if hasattr(self, 'criterion'):
            state['criterion'] = self.criterion
        
        try:
            torch.save(state, filepath)
            logger.info(f"Decision tree model saved successfully to {filepath}")
            print(f"✓ Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")

    def load_model(self, filepath: str):
        """
        Load decision tree model from file.
        
        Args:
            filepath: Path to the saved model file
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If loading fails
        """
        try:
            # Load state dictionary
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            logger.info(f"Loading decision tree model from {filepath}")
            
            # Restore tree structure
            tree_dict = state.get('tree')
            self.tree_ = self._dict_to_tree(tree_dict)
            
            # Restore core attributes
            self.is_fitted = state.get('is_fitted', False)
            self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
            self.n_features_in_ = state.get('n_features_in_')
            self.n_samples_seen_ = state.get('n_samples_seen_', 0)
            self.feature_importances_ = state.get('feature_importances_')
            
            # Restore classifier-specific attributes
            if 'classes_' in state:
                self.classes_ = state['classes_']
                self.n_classes_ = state['n_classes_']
            
            # Restore criterion
            if 'criterion' in state:
                self.criterion = state['criterion']
            
            # Restore model parameters
            params = state.get('params', {})
            for key, value in params.items():
                if hasattr(self, key) and key not in ['is_fitted', 'device', 'n_features_in_', 
                                                        'n_samples_seen_', 'classes_', 'n_classes_']:
                    setattr(self, key, value)
            
            logger.info(f"Decision tree model loaded successfully. is_fitted={self.is_fitted}")
            print(f"✓ Model loaded from {filepath}")
            
            # Verify the loaded model
            if self.is_fitted and self.tree_ is None:
                logger.warning("Model marked as fitted but tree structure is None")
                
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")


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
        """Calculate MSE or MAE impurity."""
        if len(y) == 0:
            return 0.0
        
        if self.criterion == 'mse' or self.criterion == 'squared_error':
            mean = y.mean()
            return ((y - mean) ** 2).mean().item()
        elif self.criterion == 'mae' or self.criterion == 'absolute_error':
            median = torch.median(y)
            return torch.abs(y - median).mean().item()
        else:
            # Default to MSE
            mean = y.mean()
            return ((y - mean) ** 2).mean().item()
    
    def _calculate_leaf_value(self, y: torch.Tensor) -> float:
        """Calculate mean for leaf node."""
        return y.mean().item()
    
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
        
        # Update feature info
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
        """Calculate R² score."""
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        if y.ndim > 1:
            y = y.squeeze()
        
        if isinstance(predictions, torch.Tensor):
            predictions = self._to_numpy(predictions)
        
        # Calculate R² score (coefficient of determination)
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
        
        # Update feature info
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


# Test code
# if __name__ == '__main__':
#     import numpy as np
#     from sklearn.datasets import load_iris, load_diabetes
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score, r2_score
#     import os
    
#     print("=" * 70)
#     print("TESTING DECISION TREE CLASSIFIER")
#     print("=" * 70)
    
#     # Load Iris dataset
#     iris = load_iris()
#     X, y = iris.data, iris.target
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42
#     )
    
#     print(f"\nDataset Info:")
#     print(f"  Training samples: {len(X_train)}")
#     print(f"  Test samples: {len(X_test)}")
#     print(f"  Features: {X_train.shape[1]}")
#     print(f"  Classes: {len(np.unique(y))}")
    
#     # Train classifier
#     print("\n[1] Training Decision Tree Classifier...")
#     clf = DecisionTreeClassifier(
#         max_depth=5,
#         min_samples_split=2,
#         criterion='gini',
#         random_state=42
#     )
#     clf.fit(X_train, y_train, verbose=True)
    
#     # Evaluate
#     train_pred = clf.predict(X_train)
#     test_pred = clf.predict(X_test)
    
#     train_acc = accuracy_score(y_train, train_pred)
#     test_acc = accuracy_score(y_test, test_pred)
    
#     print(f"\n[2] Model Performance:")
#     print(f"  Training Accuracy: {train_acc:.4f}")
#     print(f"  Test Accuracy: {test_acc:.4f}")
#     print(f"  Tree Depth: {clf.get_depth()}")
#     print(f"  Number of Leaves: {clf.get_n_leaves()}")
    
#     # Test predict_proba
#     probas = clf.predict_proba(X_test[:3])
#     print(f"\n[3] Sample Predictions:")
#     for i in range(3):
#         print(f"  Sample {i}: Pred={test_pred[i]}, True={y_test[i]}, " + 
#               f"Proba={probas[i].round(3)}")
    
#     # Feature importances
#     print(f"\n[4] Feature Importances:")
#     for i, imp in enumerate(clf.feature_importances_):
#         print(f"  {iris.feature_names[i]}: {imp:.4f}")
    
#     # Save model
#     save_path = 'dt_classifier_test.pth'
#     print(f"\n[5] Saving model to '{save_path}'...")
#     clf.save_model(save_path)
    
#     # Load model
#     print(f"\n[6] Loading model from '{save_path}'...")
#     clf_loaded = DecisionTreeClassifier()
#     clf_loaded.load_model(save_path)
    
#     # Verify loaded model
#     test_pred_loaded = clf_loaded.predict(X_test)
#     test_acc_loaded = accuracy_score(y_test, test_pred_loaded)
    
#     print(f"\n[7] Loaded Model Verification:")
#     print(f"  Test Accuracy: {test_acc_loaded:.4f}")
#     print(f"  Predictions match original: {np.array_equal(test_pred, test_pred_loaded)}")
#     print(f"  Tree depth preserved: {clf_loaded.get_depth() == clf.get_depth()}")
#     print(f"  Feature importances match: {np.allclose(clf.feature_importances_, clf_loaded.feature_importances_)}")
    
#     # Clean up
#     if os.path.exists(save_path):
#         os.remove(save_path)
#         print(f"\n  ✓ Cleaned up test file")
    
    
#     # TEST REGRESSOR
#     print("\n\n" + "=" * 70)
#     print("TESTING DECISION TREE REGRESSOR")
#     print("=" * 70)
    
#     # Load Diabetes dataset
#     diabetes = load_diabetes()
#     X, y = diabetes.data, diabetes.target
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42
#     )
    
#     print(f"\nDataset Info:")
#     print(f"  Training samples: {len(X_train)}")
#     print(f"  Test samples: {len(X_test)}")
#     print(f"  Features: {X_train.shape[1]}")
    
#     # Train regressor
#     print("\n[1] Training Decision Tree Regressor...")
#     reg = DecisionTreeRegressor(
#         max_depth=5,
#         min_samples_split=5,
#         criterion='mse',
#         random_state=42
#     )
#     reg.fit(X_train, y_train, verbose=True)
    
#     # Evaluate
#     train_pred = reg.predict(X_train)
#     test_pred = reg.predict(X_test)
    
#     train_r2 = r2_score(y_train, train_pred)
#     test_r2 = r2_score(y_test, test_pred)
    
#     print(f"\n[2] Model Performance:")
#     print(f"  Training R² Score: {train_r2:.4f}")
#     print(f"  Test R² Score: {test_r2:.4f}")
#     print(f"  Tree Depth: {reg.get_depth()}")
#     print(f"  Number of Leaves: {reg.get_n_leaves()}")
    
#     # Sample predictions
#     print(f"\n[3] Sample Predictions:")
#     for i in range(3):
#         print(f"  Sample {i}: Pred={test_pred[i]:.2f}, True={y_test[i]:.2f}, " +
#               f"Error={abs(test_pred[i] - y_test[i]):.2f}")
    
#     # Feature importances
#     print(f"\n[4] Top 5 Important Features:")
#     top_features = np.argsort(reg.feature_importances_)[-5:][::-1]
#     for idx in top_features:
#         print(f"  Feature {idx}: {reg.feature_importances_[idx]:.4f}")
    
#     # Save model
#     save_path = 'dt_regressor_test.pth'
#     print(f"\n[5] Saving model to '{save_path}'...")
#     reg.save_model(save_path)
    
#     # Load model
#     print(f"\n[6] Loading model from '{save_path}'...")
#     reg_loaded = DecisionTreeRegressor()
#     reg_loaded.load_model(save_path)
    
#     # Verify loaded model
#     test_pred_loaded = reg_loaded.predict(X_test)
#     test_r2_loaded = r2_score(y_test, test_pred_loaded)
    
#     print(f"\n[7] Loaded Model Verification:")
#     print(f"  Test R² Score: {test_r2_loaded:.4f}")
#     print(f"  Predictions match original: {np.allclose(test_pred, test_pred_loaded)}")
#     print(f"  Tree depth preserved: {reg_loaded.get_depth() == reg.get_depth()}")
#     print(f"  Feature importances match: {np.allclose(reg.feature_importances_, reg_loaded.feature_importances_)}")
    
#     # Clean up
#     if os.path.exists(save_path):
#         os.remove(save_path)
#         print(f"\n  ✓ Cleaned up test file")
    
#     print("\n" + "=" * 70)
#     print("ALL TESTS COMPLETED SUCCESSFULLY!")
#     print("=" * 70)
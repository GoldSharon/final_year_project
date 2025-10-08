"""
Lightning ML - PyTorch-based Machine Learning Library
GPU-accelerated classical ML algorithms with scikit-learn compatible API

Author: Gold Sharon
Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Gold Sharon"
__license__ = "MIT"

# Base classes
from .base_model import (
    BaseModel,
    BaseSupervisedModel,
    BaseUnsupervisedModel,
    BaseNeuralModel,
    BaseTreeModel
)

# Regression models
from .regression import (
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
    LassoRegression   
)

# Tree models
from .tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

# Ensemble models
from .ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    BaggingClassifier
)

# Support Vector Machines
from .svm import (
    SVMClassifier,
    SVMRegressor,
)

# K-Nearest Neighbors
from .neighbours import (
    KNNClassifier,
    KNNRegressor,
)

# Clustering
from .cluster import (
    KMeans,
    DBSCAN
)

# Association Rule Mining
from .market_basket import Apriori

# Public API - organized by category
__all__ = [
    # Base classes
    'BaseModel',
    'BaseSupervisedModel',
    'BaseUnsupervisedModel',
    'BaseNeuralModel',
    'BaseTreeModel',
    
    # Regression
    'LinearRegression',
    'LogisticRegression',
    'RidgeRegression',
    'LassoRegression'
    
    
    
    # Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    
    # Ensemble
    'RandomForestClassifier',
    'RandomForestRegressor',
    'BaggingClassifier',
    
    # SVM
    'SVMClassifier',
    'SVMRegressor',
    
    # KNN
    'KNNClassifier',
    'KNNRegressor',
    
    # Clustering
    'KMeans',
    'DBSCAN',
    
    # Association Rules
    'Apriori',
]

# Version info function
def get_version():
    """Get Lightning ML version"""
    return __version__

def get_config():
    """Get Lightning ML configuration"""
    import torch
    return {
        'version': __version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
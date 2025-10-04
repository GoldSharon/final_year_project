"""
Lightning ML - PyTorch-based Machine Learning Library
GPU-accelerated classical ML algorithms with scikit-learn compatible API
"""

__version__ = "0.1.0"
__author__ = "Gold Sharon"

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
    BaggingClassifier,
    BaggingRegressor,
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
    DBSCAN,
    AgglomerativeClustering,
)

# Association Rule Mining
from .market_basket import Apriori

# Public API
__all__ = [
    # Base classes
    'BaseModel',
    'BaseSupervisedModel',
    'BaseUnsupervisedModel',
    'BaseNeuralModel',
    'BaseTreeModel',
    
    # Regression
    'LinearRegression',
    
    # Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    
    # Ensemble
    'RandomForestClassifier',
    'RandomForestRegressor',
    'BaggingClassifier',
    'BaggingRegressor',
    
    # SVM
    'SVMClassifier',
    'SVMRegressor',
    
    # KNN
    'KNNClassifier',
    'KNNRegressor',
    
    # Clustering
    'KMeans',
    'DBSCAN',
    'AgglomerativeClustering',
    
    # Association Rules
    'Apriori',
]
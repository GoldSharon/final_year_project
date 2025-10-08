# ============================================================================
# FIXED HYPERPARAMETER SUGGESTION FUNCTIONS - ALL MODELS
# ============================================================================

def suggest_linear_regression_params(trial):
    """Linear Regression (OLS) - No tunable hyperparameters"""
    return {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'device': 'cuda'
    }


def suggest_ridge_regression_params(trial):
    """Ridge Regression (L2) - Optuna parameter suggestions"""
    return {
        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
        'epochs': trial.suggest_int('epochs', 50, 500, step=50),
        'lr': trial.suggest_float('lr', 1e-4, 0.1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        'device': 'cuda'
    }


def suggest_lasso_regression_params(trial):
    """Lasso Regression (L1) - Optuna parameter suggestions"""
    return {
        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
        'epochs': trial.suggest_int('epochs', 50, 500, step=50),
        'lr': trial.suggest_float('lr', 1e-4, 0.1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        'device': 'cuda'
    }


def suggest_logistic_regression_params(trial):
    """Logistic Regression - Optuna parameter suggestions"""
    return {
        'epochs': trial.suggest_int('epochs', 50, 500, step=50),
        'lr': trial.suggest_float('lr', 1e-4, 0.1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        'device': 'cuda'
    }


def suggest_svm_regressor_params(trial):
    """SVM Regressor - Optuna parameter suggestions"""
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    
    params = {
        'C': trial.suggest_float('C', 0.1, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        'kernel': kernel,
        'epochs': trial.suggest_int('epochs', 100, 500, step=50),
        'lr': trial.suggest_float('lr', 1e-4, 0.01, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        'device': 'cuda'
    }
    
    if kernel == 'rbf':
        gamma_type = trial.suggest_categorical('gamma_type', ['auto', 'manual'])
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto']) if gamma_type == 'auto' \
            else trial.suggest_float('gamma', 0.001, 10.0, log=True)
    elif kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
        params['gamma'] = trial.suggest_float('gamma', 0.001, 10.0, log=True)
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    elif kernel == 'sigmoid':
        params['gamma'] = trial.suggest_float('gamma', 0.001, 10.0, log=True)
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    
    return params


def suggest_svm_classifier_params(trial):
    """SVM Classifier - Optuna parameter suggestions"""
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    
    params = {
        'C': trial.suggest_float('C', 0.1, 100.0, log=True),
        'kernel': kernel,
        'epochs': trial.suggest_int('epochs', 100, 500, step=50),
        'lr': trial.suggest_float('lr', 1e-4, 0.01, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        'device': 'cuda'
    }
    
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    elif kernel == 'sigmoid':
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    
    return params


def suggest_decision_tree_regressor_params(trial):
    """Decision Tree Regressor - Optuna parameter suggestions"""
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    max_depth = trial.suggest_int('max_depth', 3, 30) if use_max_depth else None
    
    return {
        'max_depth': max_depth,
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        'criterion': trial.suggest_categorical('criterion', ['mae', 'mse']),  
        'random_state': 42,
        'device': 'cuda'
    }


def suggest_decision_tree_classifier_params(trial):
    """Decision Tree Classifier - Optuna parameter suggestions"""
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    max_depth = trial.suggest_int('max_depth', 3, 30) if use_max_depth else None
    
    return {
        'max_depth': max_depth,
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42,
        'device': 'cuda'
    }


def suggest_random_forest_regressor_params(trial):
    """Random Forest Regressor - Optuna parameter suggestions"""
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    max_depth = trial.suggest_int('max_depth', 5, 50) if use_max_depth else None
    
    # FIXED: Only suggest max_samples if bootstrap is True
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    if bootstrap:
        use_max_samples = trial.suggest_categorical('use_max_samples', [True, False])
        max_samples = trial.suggest_float('max_samples', 0.5, 1.0) if use_max_samples else None
    else:
        max_samples = None
    
    return {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500, step=10),
        'max_depth': max_depth,
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0, 0.8, 0.6]),
        'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
        'bootstrap': bootstrap,
        'max_samples': max_samples,
        'n_jobs': -1,
        'random_state': 42,
        'device': 'cuda'
    }


def suggest_random_forest_classifier_params(trial):
    """Random Forest Classifier - Optuna parameter suggestions"""
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    max_depth = trial.suggest_int('max_depth', 5, 50) if use_max_depth else None
    
    return {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500, step=10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': max_depth,
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8, 0.6]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'oob_score': False,
        'n_jobs': -1,
        'random_state': 42,
        'device': 'cuda'
    }


def suggest_bagging_classifier_params(trial):
    """Bagging Classifier - Optuna parameter suggestions"""
    return {
        'base_estimator': None,
        'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=10),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'bootstrap_features': trial.suggest_categorical('bootstrap_features', [True, False]),
        'n_jobs': -1,
        'random_state': 42,
        'device': 'cuda'
    }


def suggest_kmeans_params(trial):
    """K-Means Clustering - Optuna parameter suggestions"""
    return {
        'n_clusters': trial.suggest_int('n_clusters', 2, 20),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        'n_init': trial.suggest_int('n_init', 5, 20),
        'init': trial.suggest_categorical('init', ['k-means++', 'random']),
        'device': 'cuda'
    }


def suggest_dbscan_params(trial):
    """DBSCAN Clustering - Optuna parameter suggestions"""
    return {
        'eps': trial.suggest_float('eps', 0.1, 5.0),
        'min_samples': trial.suggest_int('min_samples', 2, 20),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine']),
        'device': 'cuda'
    }


# def suggest_agglomerative_clustering_params(trial):
#     """Agglomerative Clustering - Optuna parameter suggestions"""
#     return {
#         'n_clusters': trial.suggest_int('n_clusters', 2, 20),
#         'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single']),
#         'device': 'cuda'
#     }


def suggest_knn_classifier_params(trial):
    """KNN Classifier - Optuna parameter suggestions"""
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine']),
        'device': 'cuda'
    }


def suggest_knn_regressor_params(trial):
    """KNN Regressor - Optuna parameter suggestions"""
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine']),
        'device': 'cuda'
    }


def suggest_apriori_params(trial):
    """Apriori Algorithm - Optuna parameter suggestions"""
    use_max_length = trial.suggest_categorical('use_max_length', [True, False])
    max_length = trial.suggest_int('max_length', 2, 10) if use_max_length else None
    
    return {
        'min_support': trial.suggest_float('min_support', 0.01, 0.5),
        'min_confidence': trial.suggest_float('min_confidence', 0.1, 0.9),
        'min_lift': trial.suggest_float('min_lift', 1.0, 5.0),
        'max_length': max_length,
        'device': 'cuda'
    }
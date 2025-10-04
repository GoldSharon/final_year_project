#!/usr/bin/env python
# coding: utf-8

"""
Complete AutoML Training Pipeline
Integrates: Data Analysis → Preprocessing → AutoML Training → Evaluation
"""

import json
import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

# Import custom modules
from automl import LightningAutoML  
from cognitive_engine import CognitiveEngine
from intelligent_data_analyzer import IntelligentDataAnalyzer
from data_prep_agent import DataPrepAgent
from prompts import AUTOML_EVALUATION_SYSTEM_PROMPT


# ========================================
# Helper Functions
# ========================================

def prepare_data(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split (for classification)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    
    # Determine if stratification is possible
    stratify_param = y if stratify else None
    
    try:
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
    except ValueError:
        # Fallback if stratification fails (e.g., regression or small classes)
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )


def train_automl(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    ml_type: str = "supervised",
    method: str = "class",
    time_budget: float = 300,
    n_trials: int = 30,
    verbose: bool = True
) -> Dict:
    """
    Train AutoML models and return evaluation results.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        ml_type: "supervised" or "unsupervised"
        method: "class", "reg", "cluster", or "apriori"
        time_budget: Maximum training time in seconds
        n_trials: Number of hyperparameter optimization trials
        verbose: Print progress messages
    
    Returns:
        Dictionary containing best model info, top 3 models, and metrics
    """
    if verbose:
        print("\n" + "="*80)
        print(f"TRAINING AUTOML - {ml_type.upper()} / {method.upper()}")
        print("="*80)
    
    # Initialize AutoML
    automl = LightningAutoML(
        ml_type=ml_type,
        method=method,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_budget=time_budget,
        n_trials=n_trials,
        verbose=verbose
    )
    
    # Train models
    automl.fit()
    
    # Evaluate results
    results = automl.evaluate()
    leaderboard = automl.get_leaderboard()
    
    # Top 3 models
    top3 = leaderboard.head(3).to_dict(orient="records") if not leaderboard.empty else []
    
    # Overfitting check: compare train vs test score
    train_score = automl.best_model.score if automl.best_model else 0.0
    test_score = list(results.values())[0] if results else 0.0
    overfitted = abs(train_score - test_score) > 0.1  # heuristic threshold
    
    # Prepare JSON output
    data_json = {
        "best_model": {
            "name": automl.best_model.model_name if automl.best_model else "None",
            "hyperparameters": automl.best_model.hyperparameters if automl.best_model else {},
            "train_score": float(train_score),
            "test_score": float(test_score),
            "overfitted": bool(overfitted),
            "device": str(automl.best_model.device) if automl.best_model else "cpu"
        },
        "top3_models": top3,
        "metrics": {k: float(v) for k, v in results.items()},
        "leaderboard": leaderboard.to_dict(orient="records") if not leaderboard.empty else []
    }
    
    if verbose:
        print("\n" + "="*80)
        print("AUTOML TRAINING COMPLETE")
        print("="*80)
        print(f"Best Model: {data_json['best_model']['name']}")
        print(f"Train Score: {data_json['best_model']['train_score']:.4f}")
        print(f"Test Score: {data_json['best_model']['test_score']:.4f}")
        print(f"Overfitted: {data_json['best_model']['overfitted']}")
        print("="*80)
    
    return data_json


def evaluate_automl_with_llm(
    automl_results: Dict,
    engine: CognitiveEngine,
    dataset_info: Optional[Dict] = None
) -> Dict:
    """
    Use LLM to evaluate AutoML results and provide insights.
    
    Args:
        automl_results: Results from train_automl()
        engine: CognitiveEngine instance
        dataset_info: Optional dataset metadata
    
    Returns:
        LLM evaluation and recommendations
    """
    print("\n" + "="*80)
    print("LLM EVALUATION OF AUTOML RESULTS")
    print("="*80)
    
    # Prepare input for LLM
    evaluation_input = {
        "automl_results": automl_results,
        "dataset_info": dataset_info or {}
    }
    
    # Get LLM evaluation
    llm_response = engine.chat_llm(
        AUTOML_EVALUATION_SYSTEM_PROMPT,
        json.dumps(evaluation_input, indent=2)
    )
    
    print("\n✓ LLM Evaluation Complete")
    return llm_response


# ========================================
# Complete Training Pipeline
# ========================================

def complete_training_pipeline(
    df: pd.DataFrame,
    target_column: str,
    ml_type: str = "supervised",
    method: str = "class",
    engine: Optional[CognitiveEngine] = None,
    test_size: float = 0.2,
    time_budget: float = 300,
    n_trials: int = 30,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Complete end-to-end pipeline: Analysis → Preprocessing → AutoML → Evaluation
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        ml_type: "supervised" or "unsupervised"
        method: "class", "reg", "cluster", or "apriori"
        engine: CognitiveEngine instance (required for LLM features)
        test_size: Train/test split ratio
        time_budget: AutoML training time budget
        n_trials: Number of hyperparameter trials
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Complete results dictionary
    """
    if verbose:
        print("\n" + "="*90)
        print(" "*30 + "COMPLETE TRAINING PIPELINE")
        print("="*90)
        print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"ML Type: {ml_type} | Method: {method}")
        print(f"Target: {target_column}")
        print("="*90)
    
    # Initialize engine if not provided
    if engine is None:
        if verbose:
            print("\n⚠ Warning: No CognitiveEngine provided. Initializing default engine...")
        engine = CognitiveEngine(model_name="nchapman/dolphin3.0-llama3:3b")
    
    # ==========================================
    # PHASE 1: Data Analysis
    # ==========================================
    if verbose:
        print("\n" + "─"*90)
        print("PHASE 1: DATA ANALYSIS")
        print("─"*90)
    
    # Create complete dataframe for analysis (with target)
    df_analysis = df.copy()
    
    # Split data FIRST (important for proper preprocessing)
    feature_cols = [col for col in df.columns if col != target_column]
    
    if method in ["class", "reg"]:
        # Supervised learning
        df_train, df_test = train_test_split(
            df_analysis,
            test_size=test_size,
            random_state=random_state,
            stratify=df_analysis[target_column] if method == "class" else None
        )
    else:
        # Unsupervised learning
        df_train, df_test = train_test_split(
            df_analysis,
            test_size=test_size,
            random_state=random_state
        )
    
    # Analyze TRAINING data only
    analyzer = IntelligentDataAnalyzer(
        ml_type=ml_type,
        method=method,
        engine=engine,
        target_column=target_column if method in ["class", "reg"] else None
    )
    
    report, preprocessing_config = analyzer.analyze_dataframe(df_train)
    
    if verbose:
        print(analyzer.get_preprocessing_report(report, preprocessing_config))
    
    # ==========================================
    # PHASE 2: Data Preprocessing
    # ==========================================
    if verbose:
        print("\n" + "─"*90)
        print("PHASE 2: DATA PREPROCESSING")
        print("─"*90)
    
    # Separate features and target
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_column].copy() if method in ["class", "reg"] else None
    
    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_column].copy() if method in ["class", "reg"] else None
    
    # Preprocess training data (FIT_TRANSFORM)
    prep_agent = DataPrepAgent(
        feature_df=X_train,
        target=y_train,
        ml_type=ml_type,
        task_type=method,
        verbose=verbose
    )
    
    X_train_processed, y_train_processed = prep_agent.fit_transform(preprocessing_config)
    
    # Preprocess test data (TRANSFORM ONLY - prevents data leakage)
    X_test_processed, y_test_processed = prep_agent.transform(X_test, y_test)
    
    if verbose:
        prep_agent.print_report()
    
    # ==========================================
    # PHASE 3: AutoML Training
    # ==========================================
    if verbose:
        print("\n" + "─"*90)
        print("PHASE 3: AUTOML TRAINING")
        print("─"*90)
    
    automl_results = train_automl(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train_processed,
        y_test=y_test_processed,
        ml_type=ml_type,
        method=method,
        time_budget=time_budget,
        n_trials=n_trials,
        verbose=verbose
    )
    
    # ==========================================
    # PHASE 4: LLM Evaluation
    # ==========================================
    if verbose:
        print("\n" + "─"*90)
        print("PHASE 4: LLM EVALUATION")
        print("─"*90)
    
    dataset_info = {
        "original_shape": df.shape,
        "processed_train_shape": X_train_processed.shape,
        "processed_test_shape": X_test_processed.shape,
        "features": prep_agent.get_feature_names(),
        "target_column": target_column,
        "ml_type": ml_type,
        "method": method
    }
    
    llm_evaluation = evaluate_automl_with_llm(
        automl_results=automl_results,
        engine=engine,
        dataset_info=dataset_info
    )
    
    # ==========================================
    # PHASE 5: Save Results
    # ==========================================
    if verbose:
        print("\n" + "─"*90)
        print("PHASE 5: SAVING RESULTS")
        print("─"*90)
    
    # Save preprocessing artifacts
    prep_agent.save_processed_data(
        features_filepath=f'{method}_features_processed.csv',
        target_filepath=f'{method}_target_processed.csv',
        combined_filepath=f'{method}_data_processed.csv'
    )
    
    prep_agent.export_changes_json(f'{method}_preprocessing_changes.json')
    
    # Save AutoML results
    with open(f'{method}_automl_results.json', 'w') as f:
        json.dump(automl_results, f, indent=2)
    
    if verbose:
        print(f"✓ Saved processed data and AutoML results")
    
    # ==========================================
    # Compile Complete Results
    # ==========================================
    complete_results = {
        "pipeline_info": {
            "ml_type": ml_type,
            "method": method,
            "target_column": target_column,
            "train_test_split": f"{int((1-test_size)*100)}-{int(test_size*100)}",
            "random_state": random_state
        },
        "data_analysis": {
            "original_shape": df.shape,
            "train_shape": df_train.shape,
            "test_shape": df_test.shape,
            "data_quality_score": report.data_quality_score
        },
        "preprocessing": prep_agent.get_report(),
        "automl_results": automl_results,
        "llm_evaluation": llm_evaluation,
        "files_saved": {
            "features": f'{method}_features_processed.csv',
            "target": f'{method}_target_processed.csv',
            "combined": f'{method}_data_processed.csv',
            "changes": f'{method}_preprocessing_changes.json',
            "automl": f'{method}_automl_results.json'
        }
    }
    
    # Save complete results
    with open(f'{method}_complete_results.json', 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)
    
    if verbose:
        print("\n" + "="*90)
        print(" "*30 + "PIPELINE COMPLETE!")
        print("="*90)
        print(f"✓ Best Model: {automl_results['best_model']['name']}")
        print(f"✓ Test Score: {automl_results['best_model']['test_score']:.4f}")
        print(f"✓ All results saved to '{method}_complete_results.json'")
        print("="*90 + "\n")
    
    return complete_results


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    # Example: Iris Classification
    from sklearn.datasets import load_iris
    
    # Load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Initialize engine
    engine = CognitiveEngine()
    
    # Run complete pipeline
    results = complete_training_pipeline(
        df=df,
        target_column='species',
        ml_type='supervised',
        method='classification',
        engine=engine,
        test_size=0.2,
        time_budget=300,
        n_trials=30,
        random_state=42,
        verbose=True
    )
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ Results saved to: {results['files_saved']}")
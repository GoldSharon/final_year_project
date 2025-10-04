#!/usr/bin/env python
# coding: utf-8

"""
DataPrepAgent - Automated Data Preprocessing Agent (Enhanced)
Supports: Classification, Regression, Clustering, Association Rules (Apriori)
Features: Proper fit/transform separation, target outlier handling, data leakage prevention
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Literal
import json
import warnings

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropCorrelatedFeatures
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

warnings.filterwarnings('ignore')


class DataPrepAgent:
    """
    Automated data preprocessing agent with proper fit/transform separation.

    Key Features:
    - Handles target outliers for regression
    - Separate fit_transform() and transform() methods
    - Prevents data leakage
    - Supports supervised and unsupervised learning
    """

    VALID_TASK_TYPES = ['classification', 'regression', 'clustering', 'association']

    def __init__(
        self,
        feature_df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        ml_type: Literal['supervised', 'unsupervised'] = 'supervised',
        task_type: Literal['classification', 'regression', 'clustering', 'association'] = None,
        verbose: bool = True
    ):
        """
        Initialize DataPrepAgent.

        Args:
            feature_df: Input features DataFrame
            target: Target variable (only for supervised learning)
            ml_type: 'supervised' or 'unsupervised'
            task_type: One of ['classification', 'regression', 'clustering', 'association']
            verbose: Print progress messages
        """
        if task_type and task_type not in self.VALID_TASK_TYPES:
            raise ValueError(f"task_type must be one of {self.VALID_TASK_TYPES}")

        self.feature_df = feature_df.copy()
        self.target = target.copy() if target is not None else None
        self.ml_type = ml_type
        self.task_type = task_type
        self.verbose = verbose

        # Validate task type and target consistency
        if task_type in ['classification', 'regression'] and target is None:
            warnings.warn(f"Supervised learning ({task_type}) typically requires a target variable")
        if task_type in ['clustering', 'association'] and target is not None:
            warnings.warn(f"Unsupervised learning ({task_type}) doesn't use target, ignoring it")
            self.target = None

        # Store original data
        self.original_df = feature_df.copy()
        self.original_target = target.copy() if target is not None else None

        # Store transformers
        self.transformers = {}
        self.preprocessing_report = {}

        # Target preprocessing storage
        self.target_encoder = None
        self.target_winsorizer = None  # NEW: For regression target outliers
        self.label_mapping = {}

        # Feature engineering storage
        self.feature_engineering_code = []

        # Fit status
        self._is_fitted = False

        if self.verbose:
            print(f"DataPrepAgent initialized:")
            print(f"  • Features shape: {self.feature_df.shape}")
            print(f"  • ML type: {self.ml_type}")
            print(f"  • Task type: {self.task_type}")
            print(f"  • Has target: {self.target is not None}")
            if self.target is not None:
                print(f"  • Target type: {self.target.dtype}")
                print(f"  • Target shape: {self.target.shape}")

    def fit_transform(self, preprocessing_config: Dict) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit transformers on data and apply transformations.
        Use this method ONLY on training data.

        Args:
            preprocessing_config: Configuration dictionary from analysis agent

        Returns:
            Tuple of (preprocessed_features, preprocessed_target)
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"FITTING AND TRANSFORMING DATA - {self.task_type.upper()}")
            print("="*70)

        # Validate config
        self._validate_config(preprocessing_config)

        # Step -1: Preprocess target (FIT and TRANSFORM)
        if self.task_type in ['classification', 'regression'] and self.target is not None:
            self._fit_transform_target(preprocessing_config)

        # Step 0: Drop columns
        self._step_0_drop_columns(preprocessing_config)

        # Step 1: Handle missing values (FIT and TRANSFORM)
        self._step_1_fit_transform_missing(preprocessing_config)

        # Step 2: Handle outliers (FIT and TRANSFORM)
        if self.task_type != 'association':
            self._step_2_fit_transform_outliers(preprocessing_config)

        # Step 3: Feature engineering
        if self.task_type != 'association':
            self._step_3_feature_engineering(preprocessing_config)

        # Step 4: Encode categorical variables (FIT and TRANSFORM)
        self._step_4_fit_transform_categorical(preprocessing_config)

        # Step 5: Transform features (FIT and TRANSFORM)
        if self.task_type in ['clustering', 'regression']:
            self._step_5_fit_transform_features(preprocessing_config)

        # Step 6: Scale features (FIT and TRANSFORM)
        if self.task_type in ['clustering', 'regression', 'classification']:
            self._step_6_fit_transform_scale(preprocessing_config)

        # Step 7: Feature selection (FIT and TRANSFORM)
        self._step_7_fit_transform_feature_selection(preprocessing_config)

        # Step 8: Association-specific preprocessing
        if self.task_type == 'association':
            self._step_8_association_preprocessing(preprocessing_config)

        self._is_fitted = True

        # Generate report
        self._generate_report(preprocessing_config)

        if self.verbose:
            print("\n" + "="*70)
            print("FIT_TRANSFORM COMPLETE")
            print("="*70)
            print(f"Final shape: {self.feature_df.shape}")
            print(f"Original shape: {self.original_df.shape}")

        return self.feature_df, self.target

    def transform(self, new_df: pd.DataFrame, new_target: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform new data using fitted transformers.
        Use this method on validation/test data to prevent data leakage.

        Args:
            new_df: New features DataFrame
            new_target: New target variable (optional)

        Returns:
            Tuple of (transformed_features, transformed_target)
        """
        if not self._is_fitted:
            raise ValueError("Agent must be fitted first. Call fit_transform() on training data.")

        if self.verbose:
            print("\n" + "="*70)
            print(f"TRANSFORMING NEW DATA - {self.task_type.upper()}")
            print("="*70)
            print(f"Input shape: {new_df.shape}")

        df = new_df.copy()
        target = new_target.copy() if new_target is not None else None

        # Transform target if provided
        if target is not None and self.task_type in ['classification', 'regression']:
            target = self._transform_target(target)

        # Step 0: Drop columns
        if 'dropped_columns' in self.transformers:
            cols = [col for col in self.transformers['dropped_columns'] if col in df.columns]
            if cols:
                df.drop(columns=cols, inplace=True)
                if self.verbose:
                    print(f"  • Dropped {len(cols)} columns")

        # Step 1: Imputation
        for key in ['numeric_imputer', 'categorical_imputer']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                    if self.verbose:
                        print(f"  • Applied {key}")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Warning: Could not apply {key}: {str(e)}")

        # Step 2: Outlier handling
        if 'winsorizer' in self.transformers:
            try:
                df = self.transformers['winsorizer'].transform(df)
                if self.verbose:
                    print(f"  • Applied winsorizer")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Warning: Could not apply winsorizer: {str(e)}")

        # Step 3: Feature engineering
        if hasattr(self, 'feature_engineering_code') and self.feature_engineering_code:
            for feat_config in self.feature_engineering_code:
                new_feature = feat_config['new_feature']
                source_cols = feat_config['source_columns']
                transformation = feat_config['transformation']

                if all(col in df.columns for col in source_cols):
                    try:
                        for code in transformation:
                            exec(code)
                        if self.verbose:
                            print(f"  • Created feature: {new_feature}")
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠ Failed to create {new_feature}: {e}")

        # Step 4: Encoding
        for key in ['onehot_encoder', 'frequency_encoder']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                    if self.verbose:
                        print(f"  • Applied {key}")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Warning: Could not apply {key}: {str(e)}")

        # Step 5: Transformations
        for key in ['log_transformer', 'yeo_johnson_transformer']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                    if self.verbose:
                        print(f"  • Applied {key}")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Warning: Could not apply {key}: {str(e)}")

        # Step 6: Scaling
        if 'scaler' in self.transformers:
            try:
                df = self.transformers['scaler'].transform(df)
                if self.verbose:
                    print(f"  • Applied scaler")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Warning: Could not apply scaler: {str(e)}")

        # Step 7: Feature selection
        for key in ['constant_dropper', 'duplicate_dropper', 'correlation_dropper']:
            if key in self.transformers:
                try:
                    df = self.transformers[key].transform(df)
                    if self.verbose:
                        print(f"  • Applied {key}")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Warning: Could not apply {key}: {str(e)}")

        if self.verbose:
            print(f"  ✓ Output shape: {df.shape}")

        return df, target

    # ---------------------------
    # Target Preprocessing (NEW)
    # ---------------------------

    def _fit_transform_target(self, config: Dict):
        """Fit and transform target variable for supervised learning."""
        if self.target is None:
            return

        if self.verbose:
            print(f"\n[Target Preprocessing] Fitting and transforming target...")

        # Remove missing values
        missing_count = self.target.isnull().sum()
        if missing_count > 0:
            valid_indices = self.target.notna()
            self.target = self.target[valid_indices]
            self.feature_df = self.feature_df[valid_indices]
            if self.verbose:
                print(f"  • Removed {missing_count} rows with missing target")

        # Classification: Encode categorical target
        if self.task_type == 'classification':
            if self.target.dtype == 'object' or self.target.dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                self.target_encoder = LabelEncoder()
                self.target = pd.Series(
                    self.target_encoder.fit_transform(self.target),
                    index=self.target.index,
                    name=self.target.name
                )
                self.label_mapping = dict(enumerate(self.target_encoder.classes_))
                if self.verbose:
                    print(f"  • Encoded categorical target: {len(self.label_mapping)} classes")

        # Regression: Handle outliers in target
        elif self.task_type == 'regression':
            # Convert to numeric if needed
            if self.target.dtype == 'object':
                self.target = pd.to_numeric(self.target, errors='coerce')
                valid_indices = self.target.notna()
                invalid_count = (~valid_indices).sum()
                if invalid_count > 0:
                    self.target = self.target[valid_indices]
                    self.feature_df = self.feature_df[valid_indices]
                    if self.verbose:
                        print(f"  • Removed {invalid_count} non-numeric target values")

            # Check for extreme outliers
            Q1 = self.target.quantile(0.25)
            Q3 = self.target.quantile(0.75)
            IQR = Q3 - Q1
            extreme_outliers = ((self.target < (Q1 - 3*IQR)) | (self.target > (Q3 + 3*IQR))).sum()

            if extreme_outliers > 0:
                if self.verbose:
                    print(f"  ⚠ Found {extreme_outliers} extreme outliers in target")

                # Apply winsorization to target
                target_df = pd.DataFrame({self.target.name or 'target': self.target})
                self.target_winsorizer = Winsorizer(
                    capping_method='iqr',
                    tail='both',
                    fold=3.0,  # Use 3.0 for extreme outliers
                    variables=[self.target.name or 'target']
                )
                target_transformed = self.target_winsorizer.fit_transform(target_df)
                self.target = target_transformed[self.target.name or 'target']

                if self.verbose:
                    print(f"  • Winsorized target (3×IQR method)")
                    print(f"  • Target range: [{self.target.min():.2f}, {self.target.max():.2f}]")

        if self.verbose:
            print(f"  ✓ Target shape: {self.target.shape}")
            print(f"  ✓ Target dtype: {self.target.dtype}")

    def _transform_target(self, target: pd.Series) -> pd.Series:
        """Transform new target using fitted transformers."""
        target = target.copy()

        # Remove missing values
        valid_indices = target.notna()
        target = target[valid_indices]

        # Classification: Encode
        if self.task_type == 'classification' and self.target_encoder is not None:
            target = pd.Series(
                self.target_encoder.transform(target),
                index=target.index,
                name=target.name
            )

        # Regression: Handle outliers and convert to numeric
        elif self.task_type == 'regression':
            if target.dtype == 'object':
                target = pd.to_numeric(target, errors='coerce')
                valid_indices = target.notna()
                target = target[valid_indices]

            # Apply winsorization if fitted
            if self.target_winsorizer is not None:
                target_df = pd.DataFrame({target.name or 'target': target})
                target_transformed = self.target_winsorizer.transform(target_df)
                target = target_transformed[target.name or 'target']

        return target

    def inverse_transform_target(self, y_pred: np.ndarray, target_name: str = None) -> np.ndarray:
        """
        Inverse transform predictions back to original scale/labels.

        Args:
            y_pred: Predicted values
            target_name: Name of target variable (for regression winsorization)

        Returns:
            Original scale predictions
        """
        result = y_pred.copy()

        # Classification: Decode labels
        if self.task_type == 'classification' and self.target_encoder is not None:
            result = self.target_encoder.inverse_transform(result.astype(int))

        # Regression: Inverse winsorization (if needed)
        elif self.task_type == 'regression' and self.target_winsorizer is not None:
            # Note: Winsorizer doesn't support true inverse transform
            # The predictions are already on the winsorized scale
            # This is expected behavior for outlier handling
            if self.verbose:
                print("  ℹ Note: Predictions are on winsorized target scale")

        return result

    # ---------------------------
    # Feature Preprocessing Steps (Refactored)
    # ---------------------------

    def _validate_config(self, config: Dict):
        """Validate preprocessing configuration."""
        required_keys = ['columns_to_drop', 'columns_to_keep', 'pipeline_recommendation']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

    def _step_0_drop_columns(self, config: Dict):
        """Drop unnecessary columns."""
        columns_to_drop = config['columns_to_drop']['column_list']
        existing_cols = [col for col in columns_to_drop if col in self.feature_df.columns]
        if existing_cols:
            if self.verbose:
                print(f"\n[Step 0] Dropping {len(existing_cols)} columns...")
            self.feature_df.drop(columns=existing_cols, inplace=True)
            self.transformers['dropped_columns'] = existing_cols

    def _step_1_fit_transform_missing(self, config: Dict):
        """Fit and transform missing value imputation."""
        pipeline = config['pipeline_recommendation']
        missing_config = pipeline.get('step_1_missing_values', {})

        if self.verbose:
            print(f"\n[Step 1] Handling missing values...")

        if self.task_type == 'association':
            categorical_cols = self.feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                if self.feature_df[col].isnull().any():
                    self.feature_df[col] = self.feature_df[col].fillna('missing')

            numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_missing = [col for col in numeric_cols if self.feature_df[col].isnull().any()]
            if numeric_missing:
                strategy = missing_config.get('numeric_strategy', 'drop_rows')
                if strategy == 'drop_rows':
                    before = len(self.feature_df)
                    self.feature_df.dropna(subset=numeric_missing, inplace=True)
                    if self.verbose:
                        print(f"  • Dropped {before - len(self.feature_df)} rows")
                else:
                    imputer = MeanMedianImputer(imputation_method='median', variables=numeric_missing)
                    self.feature_df[numeric_missing] = imputer.fit_transform(self.feature_df[numeric_missing])
                    self.transformers['numeric_imputer'] = imputer
        else:
            # Numeric columns
            numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_missing = [col for col in numeric_cols if self.feature_df[col].isnull().any()]
            if numeric_missing:
                strategy = missing_config.get('numeric_strategy',
                                             'median' if self.task_type == 'clustering' else 'mean')
                imputer = MeanMedianImputer(imputation_method=strategy, variables=numeric_missing)
                self.feature_df[numeric_missing] = imputer.fit_transform(self.feature_df[numeric_missing])
                self.transformers['numeric_imputer'] = imputer
                if self.verbose:
                    print(f"  • Imputed {len(numeric_missing)} numeric columns ({strategy})")

            # Categorical columns
            categorical_cols = self.feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_missing = [col for col in categorical_cols if self.feature_df[col].isnull().any()]
            if categorical_missing:
                imputer = CategoricalImputer(imputation_method='frequent', variables=categorical_missing)
                self.feature_df[categorical_missing] = imputer.fit_transform(self.feature_df[categorical_missing])
                self.transformers['categorical_imputer'] = imputer
                if self.verbose:
                    print(f"  • Imputed {len(categorical_missing)} categorical columns")

    def _step_2_fit_transform_outliers(self, config: Dict):
        """Fit and transform outlier handling."""
        pipeline = config['pipeline_recommendation']
        outlier_config = pipeline.get('step_2_outliers', {})
        method = outlier_config.get('method', 'winsorize')
        columns = [col for col in outlier_config.get('columns_affected', []) if col in self.feature_df.columns]

        if not columns:
            return

        if self.verbose:
            print(f"\n[Step 2] Handling outliers ({method})...")

        if method == 'winsorize':
            params = outlier_config.get('parameters', {})
            capping_method = params.get('capping_method', 'iqr')
            fold = params.get('fold', 1.5)

            if self.task_type == 'clustering':
                fold = max(fold, 2.0)

            winsorizer = Winsorizer(capping_method=capping_method, tail='both', fold=fold, variables=columns)
            self.feature_df[columns] = winsorizer.fit_transform(self.feature_df[columns])
            self.transformers['winsorizer'] = winsorizer
            if self.verbose:
                print(f"  • Winsorized {len(columns)} columns")

    def _step_3_feature_engineering(self, config: Dict):
        """Create new features."""
        suggestions = config.get('feature_engineering_suggestions', [])
        if not suggestions:
            return

        if self.verbose:
            print(f"\n[Step 3] Feature engineering...")

        self.feature_engineering_code = suggestions

        for feat_config in suggestions:
            new_feature = feat_config['new_feature']
            source_cols = feat_config['source_columns']
            transformation = feat_config['transformation']

            if not all(col in self.feature_df.columns for col in source_cols):
                continue

            try:
                df = self.feature_df
                for code in transformation:
                    exec(code)
                if self.verbose:
                    print(f"  • Created: {new_feature}")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Failed: {new_feature}")

    def _step_4_fit_transform_categorical(self, config: Dict):
        """Fit and transform categorical encoding."""
        pipeline = config['pipeline_recommendation']
        encoding_config = pipeline.get('step_3_encoding', {})

        if self.verbose:
            print(f"\n[Step 4] Encoding categorical variables...")

        if self.task_type == 'association':
            categorical_cols = self.feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                self.feature_df[col] = self.feature_df[col].astype(str)
        else:
            # One-hot encoding
            onehot_cols = [col for col in encoding_config.get('onehot_columns', []) if col in self.feature_df.columns]
            if onehot_cols:
                for col in onehot_cols:
                    self.feature_df[col] = self.feature_df[col].astype(str).replace('nan', 'missing')

                drop_last = self.task_type == 'clustering'
                encoder = OneHotEncoder(variables=onehot_cols, drop_last=drop_last)
                self.feature_df = encoder.fit_transform(self.feature_df)
                self.transformers['onehot_encoder'] = encoder
                if self.verbose:
                    print(f"  • One-hot encoded {len(onehot_cols)} columns")

            # Frequency encoding
            frequency_cols = [col for col in encoding_config.get('frequency_columns', []) if col in self.feature_df.columns]
            if frequency_cols:
                for col in frequency_cols:
                    self.feature_df[col] = self.feature_df[col].astype(str)

                encoder = CountFrequencyEncoder(variables=frequency_cols, encoding_method='frequency')
                self.feature_df = encoder.fit_transform(self.feature_df)
                self.transformers['frequency_encoder'] = encoder
                if self.verbose:
                    print(f"  • Frequency encoded {len(frequency_cols)} columns")

    def _step_5_fit_transform_features(self, config: Dict):
        """Fit and transform feature transformations."""
        pipeline = config['pipeline_recommendation']
        transform_config = pipeline.get('step_4_transformation', {})

        if self.verbose:
            print(f"\n[Step 5] Transforming features...")

        # Log transformation
        log_cols = [col for col in transform_config.get('log_transform', []) if col in self.feature_df.columns]
        if log_cols:
            # Validate that columns are numeric
            numeric_log_cols = [col for col in log_cols if pd.api.types.is_numeric_dtype(self.feature_df[col])]

            if not numeric_log_cols:
                if self.verbose:
                    print(f"  ⚠ Skipping log transform (no numeric columns found)")
            else:
                try:
                    # Handle non-positive values
                    for col in numeric_log_cols:
                        if (self.feature_df[col] <= 0).any():
                            min_val = self.feature_df[col].min()
                            self.feature_df[col] = self.feature_df[col] - min_val + 1
                            if self.verbose:
                                print(f"  • Adjusted {col} for log transform (shifted by {-min_val + 1:.2f})")

                    transformer = LogTransformer(variables=numeric_log_cols)
                    self.feature_df = transformer.fit_transform(self.feature_df)
                    self.transformers['log_transformer'] = transformer
                    if self.verbose:
                        print(f"  • Log transformed {len(numeric_log_cols)} column(s)")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Log transformation failed: {str(e)}")

        # Yeo-Johnson transformation
        yeo_cols = [col for col in transform_config.get('yeo_johnson', []) if col in self.feature_df.columns]
        if yeo_cols:
            # Validate that columns are numeric
            numeric_yeo_cols = [col for col in yeo_cols if pd.api.types.is_numeric_dtype(self.feature_df[col])]

            if not numeric_yeo_cols:
                if self.verbose:
                    print(f"  ⚠ Skipping Yeo-Johnson transform (no numeric columns found)")
            else:
                try:
                    transformer = YeoJohnsonTransformer(variables=numeric_yeo_cols)
                    self.feature_df = transformer.fit_transform(self.feature_df)
                    self.transformers['yeo_johnson_transformer'] = transformer
                    if self.verbose:
                        print(f"  • Yeo-Johnson transformed {len(numeric_yeo_cols)} column(s)")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Yeo-Johnson transformation failed: {str(e)}")

    def _step_6_fit_transform_scale(self, config: Dict):
        """Fit and transform feature scaling."""
        pipeline = config['pipeline_recommendation']
        scaling_config = pipeline.get('step_5_scaling', {})
        method = scaling_config.get('method', 'standard')
        columns = [col for col in scaling_config.get('columns', []) if col in self.feature_df.columns]

        if not columns and self.task_type == 'clustering':
            columns = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            return

        if self.verbose:
            print(f"\n[Step 6] Scaling features ({method})...")

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        wrapper = SklearnTransformerWrapper(transformer=scaler, variables=columns)
        self.feature_df = wrapper.fit_transform(self.feature_df)
        self.transformers['scaler'] = wrapper
        if self.verbose:
            print(f"  • Scaled {len(columns)} columns")

    def _step_7_fit_transform_feature_selection(self, config: Dict):
      """Fit and transform feature selection."""
      pipeline = config['pipeline_recommendation']
      selection_config = pipeline.get('step_6_feature_selection', {})

      if self.verbose:
          print(f"\n[Step 7] Feature selection...")

      # Check if we have enough features for selection
      n_features = len(self.feature_df.columns)

      if n_features < 2:
          if self.verbose:
              print(f"  ⚠ Skipping feature selection (only {n_features} feature(s) remaining)")
          return

      # Drop constant features
      if selection_config.get('drop_constant', True):
          tol = 0.98 if self.task_type == 'clustering' else 0.99
          selector = DropConstantFeatures(tol=tol)
          try:
              self.feature_df = selector.fit_transform(self.feature_df)
              self.transformers['constant_dropper'] = selector
              if self.verbose and hasattr(selector, 'features_to_drop_'):
                  dropped = len(selector.features_to_drop_)
                  if dropped > 0:
                      print(f"  • Dropped {dropped} constant feature(s)")
          except Exception as e:
              if self.verbose:
                  print(f"  ⚠ Could not drop constant features: {str(e)}")

      # Re-check feature count after constant dropping
      if len(self.feature_df.columns) < 2:
          if self.verbose:
              print(f"  ⚠ Skipping remaining selection (only {len(self.feature_df.columns)} feature(s) remaining)")
          return

      # Drop duplicate features
      try:
          selector = DropDuplicateFeatures()
          self.feature_df = selector.fit_transform(self.feature_df)
          self.transformers['duplicate_dropper'] = selector
          if self.verbose and hasattr(selector, 'features_to_drop_'):
              dropped = len(selector.features_to_drop_)
              if dropped > 0:
                  print(f"  • Dropped {dropped} duplicate feature(s)")
      except Exception as e:
          if self.verbose:
              print(f"  ⚠ Could not drop duplicate features: {str(e)}")

      # Re-check feature count after duplicate dropping
      if len(self.feature_df.columns) < 2:
          if self.verbose:
              print(f"  ⚠ Skipping correlation dropping (only {len(self.feature_df.columns)} feature(s) remaining)")


    def _step_8_association_preprocessing(self, config: Dict):
        """Association rule mining preprocessing."""
        if self.task_type != 'association':
            return

        if self.verbose:
            print(f"\n[Step 8] Association preprocessing...")

        pipeline = config['pipeline_recommendation']
        assoc_config = pipeline.get('step_7_association', {})

        for col in self.feature_df.columns:
            if self.feature_df[col].dtype != 'object':
                if col in assoc_config.get('binning_columns', []):
                    n_bins = assoc_config.get('n_bins', 5)
                    self.feature_df[col] = pd.qcut(self.feature_df[col], q=n_bins,
                                                   labels=[f'{col}_bin_{i}' for i in range(n_bins)],
                                                   duplicates='drop')
                else:
                    self.feature_df[col] = self.feature_df[col].astype(str)

        min_support = assoc_config.get('min_support', 0.01)
        for col in self.feature_df.columns:
            value_counts = self.feature_df[col].value_counts(normalize=True)
            low_support_values = value_counts[value_counts < min_support].index
            if len(low_support_values) > 0:
                self.feature_df[col] = self.feature_df[col].replace(low_support_values, 'other')

    # ---------------------------
    # Reporting & Utility Methods
    # ---------------------------

    def _generate_report(self, config: Dict):
        """Generate comprehensive preprocessing report."""
        self.preprocessing_report = {
            'task_type': self.task_type,
            'original_shape': self.original_df.shape,
            'final_shape': self.feature_df.shape,
            'original_columns': self.original_df.columns.tolist(),
            'final_columns': self.feature_df.columns.tolist(),
            'columns_added': list(set(self.feature_df.columns) - set(self.original_df.columns)),
            'columns_removed': list(set(self.original_df.columns) - set(self.feature_df.columns)),
            'transformers_applied': list(self.transformers.keys()),
            'missing_values_before': int(self.original_df.isnull().sum().sum()),
            'missing_values_after': int(self.feature_df.isnull().sum().sum()),
            'config_summary': config.get('summary', {})
        }

        # Target preprocessing info
        if self.task_type in ['classification', 'regression'] and self.target is not None:
            self.preprocessing_report['target_info'] = {
                'original_dtype': str(self.original_target.dtype),
                'final_dtype': str(self.target.dtype),
                'original_shape': self.original_target.shape,
                'final_shape': self.target.shape,
                'was_encoded': self.target_encoder is not None,
                'was_winsorized': self.target_winsorizer is not None,
                'label_mapping': self.label_mapping if self.label_mapping else None,
                'missing_removed': int(self.original_target.isnull().sum())
            }

        # Task-specific metrics
        if self.task_type == 'clustering':
            scaled_count = 0
            if 'scaler' in self.transformers:
                scaler = self.transformers['scaler']
                if hasattr(scaler, 'variables'):
                    scaled_count = len(scaler.variables) if scaler.variables else 0

            self.preprocessing_report['clustering_metrics'] = {
                'numeric_features': len(self.feature_df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(self.feature_df.select_dtypes(exclude=[np.number]).columns),
                'scaled_features': scaled_count
            }
        elif self.task_type == 'association':
            self.preprocessing_report['association_metrics'] = {
                'total_items': sum(self.feature_df.nunique()),
                'average_items_per_column': self.feature_df.nunique().mean(),
                'transaction_count': len(self.feature_df)
            }

    def get_report(self) -> Dict:
        """Get preprocessing report."""
        return self.preprocessing_report

    def print_report(self):
        """Print formatted preprocessing report."""
        report = self.preprocessing_report
        print("\n" + "="*70)
        print("PREPROCESSING REPORT")
        print("="*70)
        print(f"Task Type: {report['task_type'].upper()}")
        print(f"Original Shape: {report['original_shape']}")
        print(f"Final Shape: {report['final_shape']}")
        print(f"Columns Added: {len(report['columns_added'])}")
        print(f"Columns Removed: {len(report['columns_removed'])}")
        print(f"Missing Values: {report['missing_values_before']} → {report['missing_values_after']}")
        print(f"Transformers Applied: {', '.join(report['transformers_applied'])}")

        # Target info
        if 'target_info' in report:
            print("\nTarget Variable:")
            print(f"  • Original dtype: {report['target_info']['original_dtype']}")
            print(f"  • Final dtype: {report['target_info']['final_dtype']}")
            print(f"  • Was encoded: {report['target_info']['was_encoded']}")
            print(f"  • Was winsorized: {report['target_info']['was_winsorized']}")
            if report['target_info']['label_mapping']:
                print(f"  • Label mapping: {report['target_info']['label_mapping']}")
            if report['target_info']['missing_removed'] > 0:
                print(f"  • Missing values removed: {report['target_info']['missing_removed']}")

        if 'clustering_metrics' in report:
            print("\nClustering-Specific Metrics:")
            for key, value in report['clustering_metrics'].items():
                print(f"  • {key}: {value}")

        if 'association_metrics' in report:
            print("\nAssociation-Specific Metrics:")
            for key, value in report['association_metrics'].items():
                print(f"  • {key}: {value}")

        print("="*70 + "\n")

    def save_report(self, filepath: str = 'preprocessing_report.json'):
        """Save preprocessing report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.preprocessing_report, f, indent=2)
        print(f"✓ Report saved to: {filepath}")

    def export_changes_json(self, filepath: str = 'preprocessing_changes.json'):
        """Export detailed preprocessing changes to JSON."""
        changes = {
            "preprocessing_summary": {
                "task_type": self.task_type,
                "timestamp": pd.Timestamp.now().isoformat(),
                "original_shape": list(self.original_df.shape),
                "final_shape": list(self.feature_df.shape)
            },
            "feature_changes": {
                "columns_removed": {
                    "count": len(set(self.original_df.columns) - set(self.feature_df.columns)),
                    "columns": list(set(self.original_df.columns) - set(self.feature_df.columns))
                },
                "columns_added": {
                    "count": len(set(self.feature_df.columns) - set(self.original_df.columns)),
                    "columns": list(set(self.feature_df.columns) - set(self.original_df.columns))
                },
                "columns_retained": {
                    "count": len(set(self.original_df.columns) & set(self.feature_df.columns)),
                    "columns": list(set(self.original_df.columns) & set(self.feature_df.columns))
                }
            },
            "transformations_applied": {
                "steps": list(self.transformers.keys()),
                "details": {}
            },
            "data_quality": {
                "missing_values": {
                    "before": int(self.original_df.isnull().sum().sum()),
                    "after": int(self.feature_df.isnull().sum().sum()),
                    "removed": int(self.original_df.isnull().sum().sum() - self.feature_df.isnull().sum().sum())
                },
                "rows": {
                    "before": len(self.original_df),
                    "after": len(self.feature_df),
                    "removed": len(self.original_df) - len(self.feature_df)
                }
            }
        }

        # Add transformation details
        if 'dropped_columns' in self.transformers:
            changes["transformations_applied"]["details"]["dropped_columns"] = self.transformers['dropped_columns']

        if 'numeric_imputer' in self.transformers:
            changes["transformations_applied"]["details"]["numeric_imputation"] = {
                "method": "median/mean",
                "columns": self.transformers['numeric_imputer'].variables
            }

        if 'categorical_imputer' in self.transformers:
            changes["transformations_applied"]["details"]["categorical_imputation"] = {
                "method": "mode",
                "columns": self.transformers['categorical_imputer'].variables
            }

        if 'winsorizer' in self.transformers:
            changes["transformations_applied"]["details"]["outlier_handling"] = {
                "method": "winsorization",
                "columns": self.transformers['winsorizer'].variables
            }

        if 'onehot_encoder' in self.transformers:
            changes["transformations_applied"]["details"]["onehot_encoding"] = {
                "original_columns": self.transformers['onehot_encoder'].variables,
                "encoded_columns": [col for col in self.feature_df.columns
                                   if any(orig in col for orig in self.transformers['onehot_encoder'].variables)]
            }

        if 'scaler' in self.transformers:
            scaler_type = type(self.transformers['scaler'].transformer).__name__
            changes["transformations_applied"]["details"]["scaling"] = {
                "method": scaler_type,
                "columns": self.transformers['scaler'].variables
            }

        # Target changes
        if self.task_type in ['classification', 'regression'] and self.target is not None:
            changes["target_changes"] = {
                "original_dtype": str(self.original_target.dtype),
                "final_dtype": str(self.target.dtype),
                "original_shape": list(self.original_target.shape),
                "final_shape": list(self.target.shape),
                "rows_removed": len(self.original_target) - len(self.target),
                "was_encoded": self.target_encoder is not None,
                "was_winsorized": self.target_winsorizer is not None
            }

            if self.target_encoder is not None:
                changes["target_changes"]["encoding"] = {
                    "type": "LabelEncoder",
                    "classes": self.label_mapping,
                    "n_classes": len(self.label_mapping)
                }

            if self.target_winsorizer is not None:
                changes["target_changes"]["outlier_handling"] = {
                    "method": "winsorization",
                    "fold": 3.0,
                    "capping_method": "iqr"
                }

            if self.task_type == 'classification':
                changes["target_changes"]["class_distribution"] = {
                    str(k): int(v) for k, v in self.target.value_counts().to_dict().items()
                }

        # Data type changes
        original_dtypes = self.original_df.dtypes.apply(str).to_dict()
        final_dtypes = self.feature_df.dtypes.apply(str).to_dict()

        dtype_changes = {}
        for col in set(original_dtypes.keys()) & set(final_dtypes.keys()):
            if original_dtypes[col] != final_dtypes[col]:
                dtype_changes[col] = {
                    "before": original_dtypes[col],
                    "after": final_dtypes[col]
                }

        if dtype_changes:
            changes["dtype_changes"] = dtype_changes

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(changes, f, indent=2)

        if self.verbose:
            print(f"✓ Preprocessing changes exported to: {filepath}")

        return changes

    def save_processed_data(self,
                           features_filepath: str = 'processed_features.csv',
                           target_filepath: str = 'processed_target.csv',
                           combined_filepath: str = 'processed_data.csv',
                           save_combined: bool = True):
        """Save preprocessed data to CSV files."""
        # Save features
        self.feature_df.to_csv(features_filepath, index=False)
        if self.verbose:
            print(f"✓ Features saved to: {features_filepath}")

        # Save target (if exists)
        if self.target is not None:
            self.target.to_csv(target_filepath, index=False, header=True)
            if self.verbose:
                print(f"✓ Target saved to: {target_filepath}")

            # Save combined (features + target)
            if save_combined:
                combined_df = self.feature_df.copy()
                combined_df[self.target.name if self.target.name else 'target'] = self.target.values
                combined_df.to_csv(combined_filepath, index=False)
                if self.verbose:
                    print(f"✓ Combined data saved to: {combined_filepath}")

        return {
            "features": features_filepath,
            "target": target_filepath if self.target is not None else None,
            "combined": combined_filepath if save_combined and self.target is not None else None
        }

    def get_feature_names(self) -> list:
        """Get list of final feature names after preprocessing."""
        return self.feature_df.columns.tolist()

    def get_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get preprocessed features and target."""
        return self.feature_df, self.target

    def get_target_mapping(self) -> Dict:
        """Get the label encoding mapping for classification targets."""
        return self.label_mapping

    def is_fitted(self) -> bool:
        """Check if agent has been fitted."""
        return self._is_fitted




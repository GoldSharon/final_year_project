"""
Apriori Algorithm for Association Rule Mining
PyTorch implementation for finding frequent itemsets and association rules
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Tuple, Set
from itertools import combinations
from collections import defaultdict
from .base_model import BaseUnsupervisedModel


class Apriori(BaseUnsupervisedModel):  
    """
    Apriori Algorithm for Association Rule Mining.
    
    Finds frequent itemsets and generates association rules from transactional data.
    
    Args:
        min_support: Minimum support threshold (0.0 to 1.0)
        min_confidence: Minimum confidence for rules (0.0 to 1.0)
        min_lift: Minimum lift for rules (default: 1.0)
        max_length: Maximum length of itemsets (None = no limit)
        device: Computation device
    
    Example:
        >>> apriori = Apriori(min_support=0.1, min_confidence=0.5)
        >>> apriori.fit(transactions)
        >>> rules = apriori.get_rules()
    """
    
    def __init__(self,
                 min_support: float = 0.1,
                 min_confidence: float = 0.5,
                 min_lift: float = 1.0,
                 max_length: Optional[int] = None,
                 device: Optional[torch.device] = None):
        super().__init__(device)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_length = max_length
        
        self.frequent_itemsets_ = {}  # {itemset: support}
        self.rules_ = []  # List of association rules
        self.transactions_ = None
        self.n_transactions_ = 0
    
    def _preprocess_data(self, X: Union[pd.DataFrame, np.ndarray]) -> List[Set]:
        """
        Preprocess data into transaction format.
        
        Args:
            X: Input data (DataFrame or array)
            
        Returns:
            List of sets (transactions)
        """
        transactions = []
        
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to transaction list
            for _, row in X.iterrows():
                # Get non-null values from the row
                transaction = set()
                for col in X.columns:
                    value = row[col]
                    if pd.notna(value) and value != '' and value != 0:
                        # Create item as "column=value"
                        if isinstance(value, (int, float)) and value == 1:
                            # Binary encoding: just use column name
                            transaction.add(str(col))
                        else:
                            # Categorical: use "column=value"
                            transaction.add(f"{col}={value}")
                
                if transaction:  # Only add non-empty transactions
                    transactions.append(transaction)
        
        elif isinstance(X, np.ndarray):
            # Assume binary matrix or transaction lists
            for row in X:
                transaction = set()
                if row.dtype == object:
                    # Transaction list format
                    transaction = set(str(item) for item in row if item)
                else:
                    # Binary matrix format
                    transaction = set(str(i) for i, val in enumerate(row) if val == 1)
                
                if transaction:
                    transactions.append(transaction)
        
        else:
            raise ValueError("X must be pandas DataFrame or numpy array")
        
        return transactions
    
    def _calculate_support(self, itemset: frozenset, transactions: List[Set]) -> float:
        """
        Calculate support for an itemset.
        
        Args:
            itemset: Set of items
            transactions: List of transaction sets
            
        Returns:
            Support value (proportion of transactions containing itemset)
        """
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        return count / len(transactions)
    
    def _generate_candidates(self, prev_itemsets: Set[frozenset], k: int) -> Set[frozenset]:
        """
        Generate candidate k-itemsets from (k-1)-itemsets.
        
        Args:
            prev_itemsets: Set of frequent (k-1)-itemsets
            k: Size of itemsets to generate
            
        Returns:
            Set of candidate k-itemsets
        """
        candidates = set()
        prev_list = list(prev_itemsets)
        
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                # Union of two (k-1)-itemsets
                union = prev_list[i] | prev_list[j]
                
                # Only keep if union has exactly k items
                if len(union) == k:
                    # Prune: all (k-1) subsets must be frequent
                    subsets = [frozenset(combo) for combo in combinations(union, k - 1)]
                    if all(subset in prev_itemsets for subset in subsets):
                        candidates.add(frozenset(union))
        
        return candidates
    
    def _find_frequent_itemsets(self, transactions: List[Set], 
                                verbose: bool = False) -> Dict[frozenset, float]:
        """
        Find all frequent itemsets using Apriori algorithm.
        
        Args:
            transactions: List of transaction sets
            verbose: Print progress
            
        Returns:
            Dictionary mapping itemsets to their support values
        """
        frequent_itemsets = {}
        
        # Find frequent 1-itemsets
        items = set()
        for transaction in transactions:
            items.update(transaction)
        
        if verbose:
            print(f"Finding frequent itemsets (min_support={self.min_support})...")
        
        # Calculate support for 1-itemsets
        current_itemsets = set()
        for item in items:
            itemset = frozenset([item])
            support = self._calculate_support(itemset, transactions)
            if support >= self.min_support:
                frequent_itemsets[itemset] = support
                current_itemsets.add(itemset)
        
        if verbose:
            print(f"  1-itemsets: {len(current_itemsets)} frequent")
        
        k = 2
        while current_itemsets and (self.max_length is None or k <= self.max_length):
            # Generate candidates
            candidates = self._generate_candidates(current_itemsets, k)
            
            # Calculate support and filter
            current_itemsets = set()
            for candidate in candidates:
                support = self._calculate_support(candidate, transactions)
                if support >= self.min_support:
                    frequent_itemsets[candidate] = support
                    current_itemsets.add(candidate)
            
            if verbose and current_itemsets:
                print(f"  {k}-itemsets: {len(current_itemsets)} frequent")
            
            k += 1
        
        return frequent_itemsets
    
    def _generate_rules(self, frequent_itemsets: Dict[frozenset, float],
                       verbose: bool = False) -> List[Dict]:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            frequent_itemsets: Dictionary of frequent itemsets
            verbose: Print progress
            
        Returns:
            List of rule dictionaries
        """
        rules = []
        
        if verbose:
            print(f"\nGenerating association rules (min_confidence={self.min_confidence})...")
        
        # Generate rules from itemsets with 2+ items
        for itemset, support in frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            # Generate all possible rules from this itemset
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if len(consequent) == 0:
                        continue
                    
                    # Calculate confidence
                    antecedent_support = frequent_itemsets.get(antecedent, 0)
                    if antecedent_support == 0:
                        continue
                    
                    confidence = support / antecedent_support
                    
                    if confidence >= self.min_confidence:
                        # Calculate lift
                        consequent_support = frequent_itemsets.get(consequent, 0)
                        if consequent_support > 0:
                            lift = confidence / consequent_support
                        else:
                            lift = 0.0
                        
                        if lift >= self.min_lift:
                            rule = {
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support,
                                'confidence': confidence,
                                'lift': lift,
                                'antecedent_support': antecedent_support,
                                'consequent_support': consequent_support
                            }
                            rules.append(rule)
        
        # Sort by lift (descending)
        rules.sort(key=lambda x: x['lift'], reverse=True)
        
        if verbose:
            print(f"  Generated {len(rules)} rules")
        
        return rules
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y=None,
            verbose: bool = False) -> 'Apriori':
        """
        Fit Apriori algorithm to find frequent itemsets and rules.
        
        Args:
            X: Transaction data (DataFrame or array)
            y: Ignored (for API consistency)
            verbose: Print progress
            
        Returns:
            self: Fitted model
        """
        # Preprocess data
        if verbose:
            print("Preprocessing transaction data...")
        
        self.transactions_ = self._preprocess_data(X)
        self.n_transactions_ = len(self.transactions_)
        
        if verbose:
            print(f"  Transactions: {self.n_transactions_}")
            all_items = set()
            for transaction in self.transactions_:
                all_items.update(transaction)
            print(f"  Unique items: {len(all_items)}")
        
        # Find frequent itemsets
        self.frequent_itemsets_ = self._find_frequent_itemsets(
            self.transactions_, verbose
        )
        
        # Generate association rules
        self.rules_ = self._generate_rules(self.frequent_itemsets_, verbose)
        
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        
        self.n_samples_seen_ = len(self.transactions_)
        self.is_fitted = True
        
        if verbose:
            print(f"\nApriori completed:")
            print(f"  Frequent itemsets: {len(self.frequent_itemsets_)}")
            print(f"  Association rules: {len(self.rules_)}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Not applicable for Apriori (association rule mining).
        Use get_rules() instead.
        """
        raise NotImplementedError(
            "Predict is not applicable for Apriori. Use get_rules() to get association rules."
        )
    
    def get_rules(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get association rules as a DataFrame.
        
        Args:
            top_n: Return only top N rules by lift (None = all)
            
        Returns:
            DataFrame with rules
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting rules")
        
        rules_list = self.rules_[:top_n] if top_n else self.rules_
        
        # Format rules for display
        formatted_rules = []
        for rule in rules_list:
            formatted_rules.append({
                'antecedent': ' & '.join(sorted(rule['antecedent'])),
                'consequent': ' & '.join(sorted(rule['consequent'])),
                'support': f"{rule['support']:.4f}",
                'confidence': f"{rule['confidence']:.4f}",
                'lift': f"{rule['lift']:.4f}"
            })
        
        return pd.DataFrame(formatted_rules)
    
    def get_frequent_itemsets(self, min_length: int = 1) -> pd.DataFrame:
        """
        Get frequent itemsets as a DataFrame.
        
        Args:
            min_length: Minimum itemset length to return
            
        Returns:
            DataFrame with frequent itemsets
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting itemsets")
        
        itemsets_list = []
        for itemset, support in self.frequent_itemsets_.items():
            if len(itemset) >= min_length:
                itemsets_list.append({
                    'itemset': ' & '.join(sorted(itemset)),
                    'length': len(itemset),
                    'support': f"{support:.4f}"
                })
        
        df = pd.DataFrame(itemsets_list)
        df = df.sort_values('support', ascending=False)
        return df
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> float:
        """
        Return average lift of top 10 rules as a score metric.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        if len(self.rules_) == 0:
            return 0.0
        
        top_rules = self.rules_[:min(10, len(self.rules_))]
        avg_lift = sum(rule['lift'] for rule in top_rules) / len(top_rules)
        return avg_lift
    
    def get_params(self):
        params = super().get_params()
        params.update({
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'min_lift': self.min_lift,
            'max_length': self.max_length,
            'n_frequent_itemsets': len(self.frequent_itemsets_) if self.is_fitted else 0,
            'n_rules': len(self.rules_) if self.is_fitted else 0
        })
        return params
    
    def save(self, filepath: str):
        """
        Save Apriori model to file.
        
        Args:
            filepath: Path to save model
        
        Example:
            >>> apriori.save('apriori_model.pt')
        """
        if not self.is_fitted:
            import warnings
            warnings.warn("Saving unfitted Apriori model")
        
        # Convert frozenset keys to list of tuples for serialization
        serializable_itemsets = {
            tuple(sorted(itemset)): support 
            for itemset, support in self.frequent_itemsets_.items()
        }
        
        # Convert sets in rules to lists for serialization
        serializable_rules = []
        for rule in self.rules_:
            serializable_rule = rule.copy()
            serializable_rule['antecedent'] = list(rule['antecedent'])
            serializable_rule['consequent'] = list(rule['consequent'])
            serializable_rules.append(serializable_rule)
        
        # Convert transactions to serializable format
        serializable_transactions = None
        if self.transactions_ is not None:
            serializable_transactions = [list(t) for t in self.transactions_]
        
        state = {
            'model_class': self.__class__.__name__,
            'params': self.get_params(),
            'is_fitted': self.is_fitted,
            'frequent_itemsets': serializable_itemsets,
            'rules': serializable_rules,
            'transactions': serializable_transactions,
            'n_transactions': self.n_transactions_,
            'n_features_in': self.n_features_in_,
            'n_samples_seen': self.n_samples_seen_,
            'training_history': self._training_history
        }
        
        try:
            torch.save(state, filepath)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Apriori model saved successfully to {filepath}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save Apriori model to {filepath}: {str(e)}")
            raise
    
    def load(self, filepath: str):
        """
        Load Apriori model from file.
        
        Args:
            filepath: Path to model file
        
        Example:
            >>> apriori = Apriori()
            >>> apriori.load('apriori_model.pt')
        """
        try:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Loading Apriori model from {filepath}")
            
            # Restore parameters
            params = state.get('params', {})
            self.min_support = params.get('min_support', self.min_support)
            self.min_confidence = params.get('min_confidence', self.min_confidence)
            self.min_lift = params.get('min_lift', self.min_lift)
            self.max_length = params.get('max_length', self.max_length)
            
            # Restore state
            self.is_fitted = state.get('is_fitted', False)
            self.n_transactions_ = state.get('n_transactions', 0)
            self.n_features_in_ = state.get('n_features_in', None)
            self.n_samples_seen_ = state.get('n_samples_seen', 0)
            self._training_history = state.get('training_history', {'loss': [], 'epoch': []})
            
            # Restore frequent itemsets (convert tuples back to frozensets)
            serializable_itemsets = state.get('frequent_itemsets', {})
            self.frequent_itemsets_ = {
                frozenset(itemset): support 
                for itemset, support in serializable_itemsets.items()
            }
            
            # Restore rules (convert lists back to sets)
            serializable_rules = state.get('rules', [])
            self.rules_ = []
            for rule in serializable_rules:
                restored_rule = rule.copy()
                restored_rule['antecedent'] = set(rule['antecedent'])
                restored_rule['consequent'] = set(rule['consequent'])
                self.rules_.append(restored_rule)
            
            # Restore transactions (convert lists back to sets)
            serializable_transactions = state.get('transactions', None)
            if serializable_transactions is not None:
                self.transactions_ = [set(t) for t in serializable_transactions]
            else:
                self.transactions_ = None
            
            logger.info(f"Apriori model loaded successfully. is_fitted={self.is_fitted}")
            logger.info(f"  Frequent itemsets: {len(self.frequent_itemsets_)}")
            logger.info(f"  Rules: {len(self.rules_)}")
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load Apriori model from {filepath}: {str(e)}")
            raise
        
        return self
    
    def to(self, device: Union[str, torch.device]):
        """
        Move Apriori model to specified device.
        
        Note: Apriori doesn't use PyTorch tensors for computation,
        but this method is provided for API consistency.
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        
        Returns:
            self: Model on new device
        
        Example:
            >>> apriori.to('cuda')
            >>> apriori.to(torch.device('cpu'))
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Apriori model device set to: {self.device}")
        logger.info("Note: Apriori uses CPU-based computations regardless of device setting")
        return self
    
# import pandas as pd
# import numpy as np
# import torch
# import os


# def create_sample_data():
#     """Create sample transaction data for testing."""
#     # Market basket data
#     data = {
#         'Milk': [1, 0, 1, 1, 0, 1, 1, 0],
#         'Bread': [1, 1, 1, 1, 0, 0, 1, 1],
#         'Butter': [1, 0, 1, 0, 0, 1, 1, 0],
#         'Eggs': [0, 1, 1, 1, 1, 0, 1, 1],
#         'Cheese': [0, 0, 1, 1, 0, 1, 0, 0]
#     }
#     return pd.DataFrame(data)


# def test_save_load():
#     """Test save and load functionality."""
#     print("=" * 60)
#     print("TEST 1: Save and Load Functionality")
#     print("=" * 60)
    
#     # Import the Apriori class
#     # from lightning_ml import Apriori  # Uncomment in actual use
    
#     # Create and fit original model
#     print("\n1. Creating and fitting original model...")
#     df = create_sample_data()
    
#     apriori1 = Apriori(
#         min_support=0.3,
#         min_confidence=0.6,
#         min_lift=1.0
#     )
    
#     apriori1.fit(df, verbose=True)
    
#     # Get results from original model
#     print("\n2. Original model results:")
#     rules1 = apriori1.get_rules()
#     print(f"   Number of rules: {len(rules1)}")
#     print(rules1.head())
    
#     itemsets1 = apriori1.get_frequent_itemsets()
#     print(f"\n   Number of frequent itemsets: {len(itemsets1)}")
    
#     # Save the model
#     filepath = "apriori_test_model.pt"
#     print(f"\n3. Saving model to {filepath}...")
#     apriori1.save(filepath)
#     print("   ✓ Model saved successfully")
    
#     # Create new model and load
#     print("\n4. Creating new model and loading saved state...")
#     apriori2 = Apriori()
#     apriori2.load(filepath)
#     print("   ✓ Model loaded successfully")
    
#     # Compare results
#     print("\n5. Comparing original vs loaded model:")
#     rules2 = apriori2.get_rules()
#     itemsets2 = apriori2.get_frequent_itemsets()
    
#     print(f"   Original rules: {len(rules1)}, Loaded rules: {len(rules2)}")
#     print(f"   Original itemsets: {len(itemsets1)}, Loaded itemsets: {len(itemsets2)}")
#     print(f"   is_fitted - Original: {apriori1.is_fitted}, Loaded: {apriori2.is_fitted}")
#     print(f"   n_transactions - Original: {apriori1.n_transactions_}, Loaded: {apriori2.n_transactions_}")
    
#     # Verify parameters
#     params1 = apriori1.get_params()
#     params2 = apriori2.get_params()
#     print(f"\n   Parameters match: {params1 == params2}")
    
#     # Cleanup
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"\n6. Cleaned up test file: {filepath}")
    
#     print("\n✓ TEST 1 PASSED\n")


# def test_device_movement():
#     """Test device movement (to method)."""
#     print("=" * 60)
#     print("TEST 2: Device Movement (to method)")
#     print("=" * 60)
    
#     # Create model
#     print("\n1. Creating Apriori model...")
#     apriori = Apriori(min_support=0.3, min_confidence=0.6)
#     print(f"   Initial device: {apriori.device}")
    
#     # Test moving to CPU
#     print("\n2. Moving to CPU...")
#     apriori.to('cpu')
#     print(f"   Current device: {apriori.device}")
#     assert str(apriori.device) == 'cpu', "Device should be CPU"
    
#     # Test moving to CUDA (if available)
#     if torch.cuda.is_available():
#         print("\n3. Moving to CUDA...")
#         apriori.to('cuda')
#         print(f"   Current device: {apriori.device}")
#         assert 'cuda' in str(apriori.device), "Device should be CUDA"
        
#         # Move back to CPU
#         apriori.to(torch.device('cpu'))
#         print(f"   Moved back to: {apriori.device}")
#     else:
#         print("\n3. CUDA not available, skipping CUDA test")
    
#     # Test with fitted model
#     print("\n4. Testing device movement with fitted model...")
#     df = create_sample_data()
#     apriori.fit(df, verbose=False)
    
#     apriori.to('cpu')
#     print(f"   Device after fit: {apriori.device}")
#     print(f"   Model still works: {apriori.is_fitted}")
    
#     rules = apriori.get_rules()
#     print(f"   Can still get rules: {len(rules)} rules found")
    
#     print("\n✓ TEST 2 PASSED\n")


# def test_save_before_fit():
#     """Test saving model before fitting."""
#     print("=" * 60)
#     print("TEST 3: Save Before Fit (Warning Test)")
#     print("=" * 60)
    
#     import warnings
    
#     print("\n1. Creating unfitted model...")
#     apriori = Apriori(min_support=0.2, min_confidence=0.5)
    
#     filepath = "apriori_unfitted.pt"
#     print(f"\n2. Saving unfitted model to {filepath}...")
#     print("   (Should show warning)")
    
#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter("always")
#         apriori.save(filepath)
        
#         if len(w) > 0:
#             print(f"   ✓ Warning caught: {w[0].message}")
    
#     print("\n3. Loading unfitted model...")
#     apriori2 = Apriori()
#     apriori2.load(filepath)
#     print(f"   is_fitted: {apriori2.is_fitted}")
    
#     # Cleanup
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"\n4. Cleaned up: {filepath}")
    
#     print("\n✓ TEST 3 PASSED\n")


# def test_full_workflow():
#     """Test complete workflow: fit -> save -> load -> predict."""
#     print("=" * 60)
#     print("TEST 4: Full Workflow Integration")
#     print("=" * 60)
    

    
#     # Create data
#     print("\n1. Creating transaction data...")
#     df = create_sample_data()
#     print(f"   Data shape: {df.shape}")
#     print(f"   Transactions:\n{df.head()}")
    
#     # Fit model
#     print("\n2. Fitting Apriori model...")
#     apriori = Apriori(
#         min_support=0.25,
#         min_confidence=0.5,
#         min_lift=1.0,
#         max_length=3
#     )
#     apriori.fit(df, verbose=True)
    
#     # Get and display results
#     print("\n3. Model results:")
#     rules = apriori.get_rules(top_n=5)
#     print("\n   Top 5 Association Rules:")
#     print(rules)
    
#     itemsets = apriori.get_frequent_itemsets(min_length=2)
#     print(f"\n   Frequent itemsets (length >= 2): {len(itemsets)}")
#     print(itemsets.head())
    
#     # Calculate score
#     score = apriori.score(df)
#     print(f"\n   Model score (avg lift): {score:.4f}")
    
#     # Save model
#     filepath = "apriori_workflow.pt"
#     print(f"\n4. Saving model to {filepath}...")
#     apriori.save(filepath)
    
#     # Load in new instance
#     print("\n5. Loading model in new instance...")
#     apriori_loaded = Apriori()
#     apriori_loaded.load(filepath)
    
#     # Verify loaded model works
#     print("\n6. Verifying loaded model...")
#     rules_loaded = apriori_loaded.get_rules(top_n=5)
#     score_loaded = apriori_loaded.score(df)
    
#     print(f"   Rules match: {len(rules) == len(rules_loaded)}")
#     print(f"   Score match: {score == score_loaded}")
#     print(f"   Score: {score_loaded:.4f}")
    
#     # Move to different device
#     print("\n7. Testing device movement...")
#     apriori_loaded.to('cpu')
#     rules_after_move = apriori_loaded.get_rules(top_n=5)
#     print(f"   Rules still accessible: {len(rules_after_move) == len(rules)}")
    
#     # Cleanup
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"\n8. Cleaned up: {filepath}")
    
#     print("\n✓ TEST 4 PASSED\n")


# def main():
#     """Run all tests."""
#     print("\n" + "=" * 60)
#     print("APRIORI SAVE/LOAD/TO METHODS TEST SUITE")
#     print("=" * 60 + "\n")
    
#     try:
#         # Run all tests
#         test_save_load()
#         test_device_movement()
#         test_save_before_fit()
#         test_full_workflow()
        
#         # Summary
#         print("=" * 60)
#         print("ALL TESTS PASSED SUCCESSFULLY! ✓")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"\n❌ TEST FAILED: {str(e)}")
#         import traceback
#         traceback.print_exc()
    
# if __name__ == '__main__':
#     main()


# ============================================================
# APRIORI SAVE/LOAD/TO METHODS TEST SUITE
# ============================================================

# ============================================================
# TEST 1: Save and Load Functionality
# ============================================================

# 1. Creating and fitting original model...
# Preprocessing transaction data...
#   Transactions: 8
#   Unique items: 5
# Finding frequent itemsets (min_support=0.3)...
#   1-itemsets: 5 frequent
#   2-itemsets: 6 frequent
#   3-itemsets: 2 frequent

# Generating association rules (min_confidence=0.6)...
#   Generated 16 rules

# Apriori completed:
#   Frequent itemsets: 13
#   Association rules: 16

# 2. Original model results:
#    Number of rules: 16
#            antecedent consequent support confidence    lift
# 0              Milk=1   Butter=1  0.5000     0.8000  1.6000
# 1            Butter=1     Milk=1  0.5000     1.0000  1.6000
# 2            Cheese=1     Milk=1  0.3750     1.0000  1.6000
# 3  Bread=1 & Butter=1     Milk=1  0.3750     1.0000  1.6000
# 4              Milk=1   Cheese=1  0.3750     0.6000  1.6000

#    Number of frequent itemsets: 13

# 3. Saving model to apriori_test_model.pt...
#    ✓ Model saved successfully

# 4. Creating new model and loading saved state...
#    ✓ Model loaded successfully

# 5. Comparing original vs loaded model:
#    Original rules: 16, Loaded rules: 16
#    Original itemsets: 13, Loaded itemsets: 13
#    is_fitted - Original: True, Loaded: True
#    n_transactions - Original: 8, Loaded: 8

#    Parameters match: True

# 6. Cleaned up test file: apriori_test_model.pt

# ✓ TEST 1 PASSED

# ============================================================
# TEST 2: Device Movement (to method)
# ============================================================

# 1. Creating Apriori model...
#    Initial device: cuda

# 2. Moving to CPU...
#    Current device: cpu

# 3. Moving to CUDA...
#    Current device: cuda
#    Moved back to: cpu

# 4. Testing device movement with fitted model...
#    Device after fit: cpu
#    Model still works: True
#    Can still get rules: 16 rules found

# ✓ TEST 2 PASSED

# ============================================================
# TEST 3: Save Before Fit (Warning Test)
# ============================================================

# 1. Creating unfitted model...

# 2. Saving unfitted model to apriori_unfitted.pt...
#    (Should show warning)
#    ✓ Warning caught: Saving unfitted Apriori model

# 3. Loading unfitted model...
#    is_fitted: False

# 4. Cleaned up: apriori_unfitted.pt

# ✓ TEST 3 PASSED

# ============================================================
# TEST 4: Full Workflow Integration
# ============================================================

# 1. Creating transaction data...
#    Data shape: (8, 5)
#    Transactions:
#    Milk  Bread  Butter  Eggs  Cheese
# 0     1      1       1     0       0
# 1     0      1       0     1       0
# 2     1      1       1     1       1
# 3     1      1       0     1       1
# 4     0      0       0     1       0

# 2. Fitting Apriori model...
# Preprocessing transaction data...
#   Transactions: 8
#   Unique items: 5
# Finding frequent itemsets (min_support=0.25)...
#   1-itemsets: 5 frequent
#   2-itemsets: 10 frequent
#   3-itemsets: 8 frequent

# Generating association rules (min_confidence=0.5)...
#   Generated 40 rules

# Apriori completed:
#   Frequent itemsets: 23
# 5. Loading model in new instance...

# 6. Verifying loaded model...
#    Rules match: True
#    Score match: True
#    Score: 1.6356

# 7. Testing device movement...
#    Rules still accessible: True

# 8. Cleaned up: apriori_workflow.pt

# ✓ TEST 4 PASSED

# ============================================================
# ALL TESTS PASSED SUCCESSFULLY! ✓
# ============================================================
# PS D:\Auto_ML\new_auto_ml> 
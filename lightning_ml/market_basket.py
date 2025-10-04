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
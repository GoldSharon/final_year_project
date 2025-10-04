import torch
import numpy as np
from typing import Union, Optional, Literal
from collections import defaultdict
import heapq

# Assuming base_model is imported
# from .base_model import BaseUnsupervisedModel

# ============================================================================
# K-MEANS CLUSTERING (Already mostly correct, minor optimization)
# ============================================================================

class KMeans:
    """
    K-Means Clustering with GPU acceleration.
    
    Args:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        n_init: Number of initializations
        init: Initialization method ('k-means++' or 'random')
        device: Computation device
    """
    
    def __init__(self,
                 n_clusters: int = 8,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 n_init: int = 10,
                 init: Literal['k-means++', 'random'] = 'k-means++',
                 device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.is_fitted = False
        self.n_features_in_ = None
        self.n_samples_seen_ = None
    
    def _to_tensor(self, X, dtype=torch.float32):
        if isinstance(X, torch.Tensor):
            return X.to(self.device, dtype=dtype)
        return torch.from_numpy(np.asarray(X)).to(self.device, dtype=dtype)
    
    def _to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()
    
    def _initialize_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize cluster centroids."""
        n_samples = len(X)
        
        if self.init == 'random':
            indices = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
            return X[indices].clone()
        
        else:  # k-means++
            centroids = torch.zeros(self.n_clusters, X.shape[1], device=self.device)
            first_idx = torch.randint(0, n_samples, (1,), device=self.device)
            centroids[0] = X[first_idx]
            
            for i in range(1, self.n_clusters):
                distances = torch.cdist(X, centroids[:i])
                min_distances = distances.min(dim=1)[0]
                probabilities = min_distances ** 2
                probabilities = probabilities / probabilities.sum()
                next_idx = torch.multinomial(probabilities, 1)
                centroids[i] = X[next_idx]
            
            return centroids
    
    def _assign_clusters(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Assign each point to nearest centroid."""
        distances = torch.cdist(X, centroids)
        return distances.argmin(dim=1)
    
    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Update centroids as mean of assigned points."""
        centroids = torch.zeros(self.n_clusters, X.shape[1], device=self.device)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(dim=0)
            else:
                centroids[k] = X[torch.randint(0, len(X), (1,), device=self.device)]
        
        return centroids
    
    def _calculate_inertia(self, X: torch.Tensor, labels: torch.Tensor, 
                          centroids: torch.Tensor) -> float:
        """Calculate sum of squared distances to centroids."""
        distances = torch.cdist(X, centroids)
        cluster_distances = distances[torch.arange(len(X)), labels]
        return (cluster_distances ** 2).sum().item()
    
    def _fit_single(self, X: torch.Tensor) -> tuple:
        """Single K-means run."""
        centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            centroid_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        inertia = self._calculate_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration + 1
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y=None,
            verbose: bool = False) -> 'KMeans':
        """Fit K-means clustering."""
        X = self._to_tensor(X, dtype=torch.float32)
        
        if X.dim() == 1:
            X = X.unsqueeze(1)
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        for i in range(self.n_init):
            centroids, labels, inertia, n_iter = self._fit_single(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter
            
            if verbose:
                print(f"Init {i+1}/{self.n_init}: inertia={inertia:.2f}")
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.is_fitted = True
        
        if verbose:
            print(f"K-Means converged in {self.n_iter_} iterations")
            print(f"Final inertia: {self.inertia_:.2f}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._to_tensor(X, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        
        labels = self._assign_clusters(X, self.cluster_centers_)
        return self._to_numpy(labels)
    
    def fit_predict(self, X: Union[np.ndarray, torch.Tensor], 
                    y=None) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X, y)
        return self._to_numpy(self.labels_)


# ============================================================================
# DBSCAN CLUSTERING (FIXED)
# ============================================================================

class DBSCAN:
    """
    DBSCAN Clustering with GPU support (Fixed version).
    
    Args:
        eps: Maximum distance between two samples for neighborhood
        min_samples: Minimum samples in neighborhood for core point
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        device: Computation device
    """
    
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: str = 'euclidean',
                 device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0
        self.is_fitted = False
        self.n_features_in_ = None
        self.n_samples_seen_ = None
    
    def _to_tensor(self, X, dtype=torch.float32):
        if isinstance(X, torch.Tensor):
            return X.to(self.device, dtype=dtype)
        return torch.from_numpy(np.asarray(X)).to(self.device, dtype=dtype)
    
    def _to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()
    
    def _calculate_distances_batched(self, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        """
        Calculate pairwise distances in batches to avoid OOM.
        
        Args:
            X: Data tensor
            batch_size: Batch size for distance calculation
            
        Returns:
            Distance matrix
        """
        n_samples = len(X)
        distances = torch.zeros((n_samples, n_samples), device=self.device)
        
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_X = X[i:end_i]
            
            if self.metric == 'euclidean':
                batch_distances = torch.cdist(batch_X, X, p=2)
            elif self.metric == 'manhattan':
                batch_distances = torch.cdist(batch_X, X, p=1)
            elif self.metric == 'cosine':
                batch_X_norm = batch_X / (batch_X.norm(dim=1, keepdim=True) + 1e-8)
                X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
                similarities = torch.mm(batch_X_norm, X_norm.T)
                batch_distances = 1 - similarities
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            distances[i:end_i] = batch_distances
        
        return distances
    
    def _expand_cluster(self, distances: torch.Tensor,
                       point_idx: int, 
                       neighbors: torch.Tensor,
                       cluster_id: int, 
                       labels: torch.Tensor,
                       visited: torch.Tensor) -> torch.Tensor:
        """
        Expand cluster from a core point (FIXED).
        
        Uses a list-based queue to avoid tensor membership issues.
        """
        labels[point_idx] = cluster_id
        
        # Convert to Python list for proper queue management
        seeds = neighbors.cpu().tolist()
        i = 0
        
        while i < len(seeds):
            current_point = seeds[i]
            
            if not visited[current_point]:
                visited[current_point] = True
                
                # Find neighbors of current point
                neighbor_mask = distances[current_point] <= self.eps
                current_neighbors = torch.where(neighbor_mask)[0]
                
                # If current point is also a core point
                if len(current_neighbors) >= self.min_samples:
                    # Add new neighbors to queue
                    for neighbor in current_neighbors.cpu().tolist():
                        if neighbor not in seeds:
                            seeds.append(neighbor)
            
            # Assign to cluster if not already assigned
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            
            i += 1
        
        return labels
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y=None,
            verbose: bool = False) -> 'DBSCAN':
        """Fit DBSCAN clustering."""
        X = self._to_tensor(X, dtype=torch.float32)
        
        if X.dim() == 1:
            X = X.unsqueeze(1)
        
        n_samples = len(X)
        
        # For large datasets, warn about memory usage
        if n_samples > 5000 and verbose:
            print(f"Warning: Computing distance matrix for {n_samples} samples may use significant memory")
        
        if verbose:
            print("Calculating distance matrix...")
        
        distances = self._calculate_distances_batched(X)
        
        # Initialize
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        visited = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        
        cluster_id = 0
        core_samples = []
        
        if verbose:
            print(f"Finding clusters (eps={self.eps}, min_samples={self.min_samples})...")
        
        for i in range(n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            
            # Find neighbors
            neighbor_mask = distances[i] <= self.eps
            neighbors = torch.where(neighbor_mask)[0]
            
            # Check if core point
            if len(neighbors) >= self.min_samples:
                core_samples.append(i)
                labels = self._expand_cluster(
                    distances, i, neighbors, cluster_id, labels, visited
                )
                cluster_id += 1
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{n_samples} points")
        
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = cluster_id
        self.n_noise_ = (labels == -1).sum().item()
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.is_fitted = True
        
        if verbose:
            print(f"\nDBSCAN completed:")
            print(f"  Clusters found: {self.n_clusters_}")
            print(f"  Core samples: {len(self.core_sample_indices_)}")
            print(f"  Noise points: {self.n_noise_}")
        
        return self
    
    def fit_predict(self, X: Union[np.ndarray, torch.Tensor], 
                    y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, y)
        return self._to_numpy(self.labels_)


# ============================================================================
# AGGLOMERATIVE CLUSTERING (OPTIMIZED)
# ============================================================================

class AgglomerativeClustering:
    """
    Agglomerative Hierarchical Clustering (Optimized version).
    
    Uses a priority queue for efficient cluster merging.
    
    Args:
        n_clusters: Number of clusters to find
        linkage: Linkage criterion ('single', 'complete', 'average', 'ward')
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        device: Computation device
    """
    
    def __init__(self,
                 n_clusters: int = 2,
                 linkage: Literal['single', 'complete', 'average', 'ward'] = 'ward',
                 metric: str = 'euclidean',
                 device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        
        self.labels_ = None
        self.n_leaves_ = None
        self.children_ = None
        self.is_fitted = False
        self.n_features_in_ = None
        self.n_samples_seen_ = None
    
    def _to_tensor(self, X, dtype=torch.float32):
        if isinstance(X, torch.Tensor):
            return X.to(self.device, dtype=dtype)
        return torch.from_numpy(np.asarray(X)).to(self.device, dtype=dtype)
    
    def _to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()
    
    def _calculate_distance(self, point1: torch.Tensor, point2: torch.Tensor) -> float:
        """Calculate distance between two points."""
        if self.metric == 'euclidean':
            return torch.sqrt(torch.sum((point1 - point2) ** 2)).item()
        elif self.metric == 'manhattan':
            return torch.sum(torch.abs(point1 - point2)).item()
        elif self.metric == 'cosine':
            return (1 - torch.dot(point1, point2) / 
                   (torch.norm(point1) * torch.norm(point2) + 1e-8)).item()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _calculate_cluster_distance(self, 
                                   cluster_a_indices: list, 
                                   cluster_b_indices: list,
                                   X: torch.Tensor) -> float:
        """
        Calculate distance between clusters (OPTIMIZED & FIXED).
        """
        if self.linkage == 'single':
            min_dist = float('inf')
            for i in cluster_a_indices:
                for j in cluster_b_indices:
                    dist = self._calculate_distance(X[i], X[j])
                    if dist < min_dist:
                        min_dist = dist
            return min_dist
        
        elif self.linkage == 'complete':
            max_dist = 0.0
            for i in cluster_a_indices:
                for j in cluster_b_indices:
                    dist = self._calculate_distance(X[i], X[j])
                    if dist > max_dist:
                        max_dist = dist
            return max_dist
        
        elif self.linkage == 'average':
            total_dist = 0.0
            count = 0
            for i in cluster_a_indices:
                for j in cluster_b_indices:
                    total_dist += self._calculate_distance(X[i], X[j])
                    count += 1
            return total_dist / count if count > 0 else 0.0
        
        elif self.linkage == 'ward':
            # Ward's method: calculate increase in sum of squared distances
            n_a = len(cluster_a_indices)
            n_b = len(cluster_b_indices)
            
            centroid_a = X[cluster_a_indices].mean(dim=0)
            centroid_b = X[cluster_b_indices].mean(dim=0)
            
            # Distance between centroids weighted by cluster sizes
            centroid_dist = torch.sum((centroid_a - centroid_b) ** 2).item()
            ward_dist = (n_a * n_b) / (n_a + n_b) * centroid_dist
            
            return ward_dist
        
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y=None,
            verbose: bool = False) -> 'AgglomerativeClustering':
        """Fit agglomerative clustering (OPTIMIZED)."""
        X = self._to_tensor(X, dtype=torch.float32)
        
        if X.dim() == 1:
            X = X.unsqueeze(1)
        
        n_samples = len(X)
        
        # Initialize clusters (each point is its own cluster)
        clusters = {i: [i] for i in range(n_samples)}
        active_clusters = set(range(n_samples))
        
        # Initialize priority queue with all pairwise distances
        if verbose:
            print("Initializing cluster distances...")
        
        heap = []
        cluster_distances = {}
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self._calculate_cluster_distance([i], [j], X)
                heapq.heappush(heap, (dist, i, j))
                cluster_distances[(i, j)] = dist
        
        self.children_ = []
        
        if verbose:
            print(f"Starting hierarchical clustering ({self.linkage} linkage)...")
        
        merge_count = 0
        n_merges = n_samples - self.n_clusters
        
        while len(active_clusters) > self.n_clusters and heap:
            # Get closest pair
            dist, i, j = heapq.heappop(heap)
            
            # Skip if clusters already merged
            if i not in active_clusters or j not in active_clusters:
                continue
            
            # Merge clusters
            new_cluster_id = n_samples + merge_count
            clusters[new_cluster_id] = clusters[i] + clusters[j]
            
            # Record merge
            self.children_.append([i, j])
            
            # Update active clusters
            active_clusters.remove(i)
            active_clusters.remove(j)
            active_clusters.add(new_cluster_id)
            
            # Calculate distances to new cluster
            for other_id in list(active_clusters):
                if other_id != new_cluster_id:
                    new_dist = self._calculate_cluster_distance(
                        clusters[new_cluster_id], 
                        clusters[other_id], 
                        X
                    )
                    heapq.heappush(heap, (new_dist, min(new_cluster_id, other_id), 
                                         max(new_cluster_id, other_id)))
            
            merge_count += 1
            
            if verbose and (merge_count % 10 == 0 or merge_count == n_merges):
                print(f"Merge {merge_count}/{n_merges}: {len(active_clusters)} clusters remaining")
        
        # Assign final labels
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        for cluster_id, (label, cluster_indices) in enumerate(
            zip(range(len(active_clusters)), 
                [clusters[cid] for cid in active_clusters])):
            for idx in cluster_indices:
                labels[idx] = label
        
        self.labels_ = labels
        self.n_leaves_ = n_samples
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.is_fitted = True
        
        if verbose:
            print(f"\nHierarchical clustering completed:")
            print(f"  Final clusters: {self.n_clusters}")
            print(f"  Samples: {n_samples}")
        
        return self
    
    def fit_predict(self, X: Union[np.ndarray, torch.Tensor], 
                    y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, y)
        return self._to_numpy(self.labels_)
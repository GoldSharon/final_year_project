import torch
import numpy as np
from typing import Union, Optional, Literal, Dict, Any
import warnings

# Import base unsupervised model for API compatibility
from .base_model import BaseUnsupervisedModel
import logging


# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

class KMeans(BaseUnsupervisedModel):
    """
    K-Means Clustering with GPU acceleration and model persistence.
    Inherits BaseUnsupervisedModel to be compatible with LightningAutoML.
    """

    def __init__(self,
                 n_clusters: int = 8,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 n_init: int = 10,
                 init: Literal['k-means++', 'random'] = 'k-means++',
                 device: Optional[torch.device] = None):
        super().__init__(device)
        # CRITICAL: store all hyperparameters as instance variables
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.init = str(init)

        # State (will be set after fit)
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

    def _update_fit_info(self, X: torch.Tensor):
        # Required by BaseUnsupervisedModel contract
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]

    def _initialize_centroids(self, X: torch.Tensor) -> torch.Tensor:
        n_samples = X.shape[0]

        if self.init == 'random':
            indices = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
            return X[indices].clone()

        # k-means++
        centroids = torch.empty((self.n_clusters, X.shape[1]), device=self.device, dtype=X.dtype)
        first_idx = torch.randint(0, n_samples, (1,), device=self.device)
        centroids[0] = X[first_idx].squeeze(0)

        for i in range(1, self.n_clusters):
            distances = torch.cdist(X, centroids[:i])  # (n_samples, i)
            min_distances = distances.min(dim=1)[0]  # (n_samples,)
            probs = (min_distances ** 2)
            total = probs.sum()
            if total.item() == 0.0:
                # fallback: uniform sampling
                next_idx = torch.randint(0, n_samples, (1,), device=self.device)
            else:
                probs = probs / total
                next_idx = torch.multinomial(probs, 1)
            centroids[i] = X[next_idx].squeeze(0)

        return centroids

    def _assign_clusters(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(X, centroids)
        return distances.argmin(dim=1)

    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centroids = torch.empty((self.n_clusters, X.shape[1]), device=self.device, dtype=X.dtype)

        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum().item() > 0:
                centroids[k] = X[mask].mean(dim=0)
            else:
                # Reinitialize empty cluster to a random sample
                centroids[k] = X[torch.randint(0, len(X), (1,), device=self.device)].squeeze(0)

        return centroids

    def _calculate_inertia(self, X: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor) -> float:
        distances = torch.cdist(X, centroids)
        cluster_distances = distances[torch.arange(len(X), device=self.device), labels]
        return (cluster_distances ** 2).sum().item()

    def _fit_single(self, X: torch.Tensor) -> tuple:
        centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            centroid_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if centroid_shift.item() < self.tol:
                return centroids, labels, self._calculate_inertia(X, labels, centroids), iteration + 1

        # reached max_iter
        labels = self._assign_clusters(X, centroids)
        return centroids, labels, self._calculate_inertia(X, labels, centroids), self.max_iter

    def fit(self, X: Union[np.ndarray, torch.Tensor], y=None, verbose: bool = False) -> 'KMeans':
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
                best_centroids = centroids.clone()
                best_labels = labels.clone()
                best_n_iter = n_iter

            if verbose:
                print(f"Init {i+1}/{self.n_init}: inertia={inertia:.6f}")

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        self._update_fit_info(X)
        self.is_fitted = True

        if verbose:
            print(f"KMeans: converged in {self.n_iter_} iterations; inertia={self.inertia_:.6f}")

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._to_tensor(X, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(1)

        labels = self._assign_clusters(X, self.cluster_centers_)
        return self._to_numpy(labels)

    def fit_predict(self, X: Union[np.ndarray, torch.Tensor], y=None) -> np.ndarray:
        self.fit(X, y)
        return self._to_numpy(self.labels_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'n_init': self.n_init,
            'init': self.init,
            'device': str(self.device)
        }
        return params.copy() if deep else params

    def save_model(self, filepath: str):
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")

        state = {
            'params': self.get_params(),
            'cluster_centers_': self.cluster_centers_.cpu() if self.cluster_centers_ is not None else None,
            'labels_': self.labels_.cpu() if self.labels_ is not None else None,
            'inertia_': self.inertia_,
            'n_iter_': self.n_iter_,
            'is_fitted': self.is_fitted
        }

        torch.save(state, filepath)

    def load_model(self, filepath: str):
        state = torch.load(filepath, map_location=self.device)
        params = state.get('params', {})

        # restore hyperparameters
        self.n_clusters = int(params.get('n_clusters', self.n_clusters))
        self.max_iter = int(params.get('max_iter', self.max_iter))
        self.tol = float(params.get('tol', self.tol))
        self.n_init = int(params.get('n_init', self.n_init))
        self.init = str(params.get('init', self.init))

        # restore fitted attributes
        cc = state.get('cluster_centers_')
        lbls = state.get('labels_')
        self.cluster_centers_ = cc.to(self.device) if isinstance(cc, torch.Tensor) else (torch.tensor(cc, device=self.device) if cc is not None else None)
        self.labels_ = lbls.to(self.device) if isinstance(lbls, torch.Tensor) else (torch.tensor(lbls, device=self.device, dtype=torch.long) if lbls is not None else None)
        self.inertia_ = state.get('inertia_')
        self.n_iter_ = state.get('n_iter_', 0)
        self.is_fitted = bool(state.get('is_fitted', False))

        # update fit info if possible
        if self.cluster_centers_ is not None and self.labels_ is not None:
            self._update_fit_info(self.cluster_centers_)  # best-effort

        return self

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device) if isinstance(device, (str, torch.device)) else device
        if self.cluster_centers_ is not None:
            self.cluster_centers_ = self.cluster_centers_.to(self.device)
        if self.labels_ is not None and isinstance(self.labels_, torch.Tensor):
            self.labels_ = self.labels_.to(self.device)
        return self


# ============================================================================
# DBSCAN
# ============================================================================

class DBSCAN(BaseUnsupervisedModel):
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: str = 'euclidean',
                 device: Optional[torch.device] = None):
        super().__init__(device)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = str(metric)

        self.labels_ = None
        self.core_sample_indices_ = None
        
        # --- FIX: Add attributes needed for predict ---
        self.components_ = None # Stores the data points of core samples
        self.component_labels_ = None # Stores the labels of core samples
        
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

    def _update_fit_info(self, X: torch.Tensor):
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]

    def _calculate_distances_batched(self, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        n_samples = X.shape[0]
        distances = torch.empty((n_samples, n_samples), device=self.device, dtype=X.dtype)

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
        labels[point_idx] = cluster_id

        seeds = [int(x) for x in neighbors.cpu().tolist()]
        i = 0

        while i < len(seeds):
            current_point = seeds[i]

            if not bool(visited[current_point].item() if isinstance(visited[current_point], torch.Tensor) else visited[current_point]):
                visited[current_point] = True

                neighbor_mask = distances[current_point] <= self.eps
                current_neighbors = torch.where(neighbor_mask)[0]

                if current_neighbors.shape[0] >= self.min_samples:
                    for neighbor in current_neighbors.cpu().tolist():
                        if int(neighbor) not in seeds:
                            seeds.append(int(neighbor))

            if int(labels[current_point].item()) == -1:
                labels[current_point] = cluster_id

            i += 1

        return labels

    def fit(self, X: Union[np.ndarray, torch.Tensor], y=None, verbose: bool = False) -> 'DBSCAN':
        X = self._to_tensor(X, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(1)

        n_samples = X.shape[0]

        if n_samples > 5000 and verbose:
            print(f"Warning: Computing distance matrix for {n_samples} samples may use significant memory")
        
        distances = self._calculate_distances_batched(X)
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        visited = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        cluster_id = 0
        core_samples_indices_list = []

        for i in range(n_samples):
            if bool(visited[i].item()):
                continue
            visited[i] = True
            neighbor_mask = distances[i] <= self.eps
            neighbors = torch.where(neighbor_mask)[0]

            if neighbors.shape[0] >= self.min_samples:
                core_samples_indices_list.append(int(i))
                labels = self._expand_cluster(distances, i, neighbors, cluster_id, labels, visited)
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = torch.tensor(core_samples_indices_list, dtype=torch.long, device=self.device)
        
        # --- FIX: Store core samples and their labels for prediction ---
        if self.core_sample_indices_.numel() > 0:
            self.components_ = X[self.core_sample_indices_]
            self.component_labels_ = self.labels_[self.core_sample_indices_]
        
        self.n_clusters_ = int(cluster_id)
        self.n_noise_ = int((labels == -1).sum().item())
        self._update_fit_info(X)
        self.is_fitted = True

        if verbose:
            print(f"\nDBSCAN completed: Clusters={self.n_clusters_}, Noise points={self.n_noise_}")

        return self

    # --- FIX: Implemented the missing predict method ---
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        # If there are no core samples, all new points are noise
        if self.components_ is None or self.components_.shape[0] == 0:
            return np.full(X.shape[0] if not isinstance(X, (int, float)) else 1, -1)

        X_tensor = self._to_tensor(X, dtype=torch.float32)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0) if len(X_tensor.shape) == 1 and X_tensor.shape[0] > 1 else X_tensor.unsqueeze(1)
        
        # Calculate distances from new points to core samples
        distances = torch.cdist(X_tensor, self.components_)

        # Find the index and value of the closest core sample for each new point
        min_distances, closest_core_indices = torch.min(distances, dim=1)

        # Assign labels, defaulting to noise (-1)
        labels = torch.full((X_tensor.shape[0],), -1, dtype=torch.long, device=self.device)

        # Find which points are within eps of a core sample
        mask = min_distances <= self.eps

        # Assign the label of the closest core sample to those points
        labels[mask] = self.component_labels_[closest_core_indices[mask]]

        return self._to_numpy(labels)

    def fit_predict(self, X: Union[np.ndarray, torch.Tensor], y=None) -> np.ndarray:
        self.fit(X, y)
        return self._to_numpy(self.labels_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'device': str(self.device)
        }
        return params.copy() if deep else params

    def save_model(self, filepath: str):
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")

        state = {
            'params': self.get_params(),
            'labels_': self.labels_.cpu() if isinstance(self.labels_, torch.Tensor) else self.labels_,
            'core_sample_indices_': self.core_sample_indices_.cpu() if isinstance(self.core_sample_indices_, torch.Tensor) else self.core_sample_indices_,
            # --- FIX: Save new attributes ---
            'components_': self.components_.cpu() if self.components_ is not None else None,
            'component_labels_': self.component_labels_.cpu() if self.component_labels_ is not None else None,
            'n_clusters_': self.n_clusters_,
            'n_noise_': self.n_noise_,
            'is_fitted': self.is_fitted
        }
        torch.save(state, filepath)

    def load_model(self, filepath: str):
        state = torch.load(filepath, map_location=self.device)
        params = state.get('params', {})

        self.eps = float(params.get('eps', self.eps))
        self.min_samples = int(params.get('min_samples', self.min_samples))
        self.metric = str(params.get('metric', self.metric))

        # --- FIX: Load new attributes ---
        for attr_name in ['labels_', 'core_sample_indices_', 'components_', 'component_labels_']:
            tensor_data = state.get(attr_name)
            if tensor_data is not None:
                setattr(self, attr_name, tensor_data.to(self.device))
        
        self.n_clusters_ = int(state.get('n_clusters_', 0))
        self.n_noise_ = int(state.get('n_noise_', 0))
        self.is_fitted = bool(state.get('is_fitted', False))
        return self

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device) if isinstance(device, (str, torch.device)) else device
        for attr_name in ['labels_', 'core_sample_indices_', 'components_', 'component_labels_']:
            attr = getattr(self, attr_name, None)
            if attr is not None and isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(self.device))
        return self

# ============================================================================
# AGGLOMERATIVE CLUSTERING ( It is very slow , please dont run it )
# ============================================================================

# logger = logging.getLogger(__name__)


# class AgglomerativeClustering(BaseUnsupervisedModel):
#     """
#     Hierarchical Agglomerative Clustering model compatible with Lightning ML framework.
#     """

#     def __init__(self, n_clusters: int = 2, linkage: str = "ward", device: Optional[torch.device] = None):
#         super().__init__(device)
#         self.n_clusters = n_clusters
#         self.linkage = linkage
#         self.labels_ = None
#         self.is_fitted = False

#     # -------------------------------------------------------------------------
#     def fit(self, X: Union[np.ndarray, torch.Tensor], y=None, **kwargs):
#         X = self._to_tensor(X)
#         self._validate_input(X)
#         n_samples = X.shape[0]
#         dists = torch.cdist(X, X, p=2)
#         clusters = {i: [i] for i in range(n_samples)}
#         active = set(range(n_samples))

#         sq_norms = (X ** 2).sum(dim=1, keepdim=True) if self.linkage == "ward" else None

#         while len(active) > self.n_clusters:
#             i, j = self._find_closest_clusters(dists, active)
#             self._merge_clusters(i, j, clusters, active, dists, X, sq_norms)

#         # Assign labels
#         label_map = {cid: idx for idx, cid in enumerate(active)}
#         self.labels_ = torch.zeros(n_samples, dtype=torch.long, device=self.device)
#         for cid, members in clusters.items():
#             self.labels_[members] = label_map[cid]

#         self._update_fit_info(X)
#         self.is_fitted = True
#         return self

#     # -------------------------------------------------------------------------
#     def _find_closest_clusters(self, dists, active):
#         min_dist = float("inf")
#         best_pair = (None, None)
#         active_list = list(active)
#         for i in range(len(active_list)):
#             for j in range(i + 1, len(active_list)):
#                 a, b = active_list[i], active_list[j]
#                 if dists[a, b] < min_dist:
#                     min_dist = dists[a, b].item()
#                     best_pair = (a, b)
#         return best_pair

#     def _merge_clusters(self, i, j, clusters, active, dists, X, sq_norms):
#         clusters[i].extend(clusters[j])
#         del clusters[j]
#         active.remove(j)
#         for k in active:
#             if k == i:
#                 continue
#             if self.linkage == "single":
#                 new_dist = torch.min(dists[i, k], dists[j, k])
#             elif self.linkage == "complete":
#                 new_dist = torch.max(dists[i, k], dists[j, k])
#             elif self.linkage == "average":
#                 ni, nj = len(clusters[i]), len(clusters[k])
#                 new_dist = (ni * dists[i, k] + nj * dists[j, k]) / (ni + nj)
#             elif self.linkage == "ward":
#                 Xi, Xk = X[clusters[i]], X[clusters[k]]
#                 mean_i, mean_k = Xi.mean(dim=0), Xk.mean(dim=0)
#                 new_dist = torch.norm(mean_i - mean_k)
#             else:
#                 raise ValueError(f"Unsupported linkage: {self.linkage}")
#             dists[i, k] = dists[k, i] = new_dist

#     # -------------------------------------------------------------------------
#     def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
#         if not self.is_fitted:
#             raise RuntimeError("Model not fitted. Call `fit` first.")
#         return self._to_numpy(self.labels_)

#     # -------------------------------------------------------------------------
#     def get_params(self):
#         params = super().get_params()
#         params.update({"n_clusters": self.n_clusters, "linkage": self.linkage})
#         return params

#     # -------------------------------------------------------------------------
#     def save_model(self, path: str):
#         if not self.is_fitted:
#             warnings.warn("Saving an unfitted model.")
#         state = {
#             "params": self.get_params(),
#             "labels_": self.labels_.cpu() if self.labels_ is not None else None,
#             "is_fitted": self.is_fitted,
#         }
#         torch.save(state, path)

#     def load_model(self, path: str):
#         state = torch.load(path, map_location=self.device)
#         params = state.get("params", {})
#         self.n_clusters = int(params.get("n_clusters", self.n_clusters))
#         self.linkage = str(params.get("linkage", self.linkage))
#         lbls = state.get("labels_")
#         self.labels_ = lbls.to(self.device) if isinstance(lbls, torch.Tensor) else torch.tensor(lbls, device=self.device)
#         self.is_fitted = bool(state.get("is_fitted", True))
#         return self

#     # -------------------------------------------------------------------------
#     def to(self, device: Union[str, torch.device]):
#         self.device = torch.device(device)
#         if self.labels_ is not None:
#             self.labels_ = self.labels_.to(self.device)
#         return self


# if __name__ == "__main__":
#     import numpy as np
#     from sklearn.datasets import make_blobs
#     import os

#     # Generate synthetic dataset
#     X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

#     # =====================
#     # KMeans Test
#     # =====================
#     print("\n=== Testing KMeans ===")
#     kmeans = KMeans(n_clusters=3)
#     kmeans.fit(X)
#     print("Params before save:", kmeans.get_params())
#     preds_before = kmeans.predict(X)

#     # Save model
#     kmeans_path = "kmeans_test.pth"
#     kmeans.save_model(kmeans_path)

#     # Create a new instance and load
#     kmeans_loaded = KMeans(n_clusters=3)
#     kmeans_loaded.load_model(kmeans_path)
#     preds_after = kmeans_loaded.predict(X)

#     # Check if predictions are identical
#     assert np.all(preds_before == preds_after), "KMeans predictions mismatch after loading"
#     print("KMeans save/load test passed ✅")

#     # Cleanup
#     os.remove(kmeans_path)

#     # =====================
#     # DBSCAN Test
#     # =====================
#     print("\n=== Testing DBSCAN ===")
#     dbscan = DBSCAN(eps=0.5, min_samples=5)
#     dbscan.fit(X)
#     print("Params before save:", dbscan.get_params())
#     preds_before = dbscan.predict(X)

#     # Save model
#     dbscan_path = "dbscan_test.pth"
#     dbscan.save_model(dbscan_path)

#     # Load
#     dbscan_loaded = DBSCAN(eps=0.5)
#     dbscan_loaded.load_model(dbscan_path)
#     preds_after = dbscan_loaded.predict(X)

#     assert np.all(preds_before == preds_after), "DBSCAN predictions mismatch after loading"
#     print("DBSCAN save/load test passed ✅")

#     os.remove(dbscan_path)

#     # =====================
#     # Agglomerative Clustering Test
#     # =====================
#     print("\n=== Testing Agglomerative Clustering ===")
#     agglom = AgglomerativeClustering(n_clusters=3)
#     agglom.fit(X)
#     print("Params before save:", agglom.get_params())
#     preds_before = agglom.predict(X)

#     # Save model
#     agglom_path = "agglo_test.pth"
#     agglom.save_model(agglom_path)

#     # Load
#     agglom_loaded = AgglomerativeClustering(n_clusters=3)
#     agglom_loaded.load_model(agglom_path)
#     preds_after = agglom_loaded.predict(X)

#     assert np.all(preds_before == preds_after), "Agglomerative predictions mismatch after loading"
#     print("Agglomerative save/load test passed ✅")

#     os.remove(agglom_path)

#     print("\nAll save/load/get_params tests completed successfully ✅")


# PS D:\Auto_ML\new_auto_ml> & C:/Users/Smile/anaconda3/envs/tensorflow/python.exe d:/Auto_ML/new_auto_ml/lightning_ml/cluster.py

# === Testing KMeans ===
# Params before save: {'n_clusters': 3, 'max_iter': 300, 'tol': 0.0001, 'n_init': 10, 'init': 'k-means++', 'device': 'cuda'}
# KMeans save/load test passed ✅

# === Testing DBSCAN ===
# Params before save: {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean', 'device': 'cuda'}
# DBSCAN save/load test passed ✅

# === Testing Agglomerative Clustering ===
# Params before save: {'device': 'cuda', 'is_fitted': True, 'n_clusters': 3, 'linkage': 'ward'}
# Agglomerative save/load test passed ✅

# All save/load/get_params tests completed successfully ✅
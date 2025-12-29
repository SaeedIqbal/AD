import numpy as np
import logging
from typing import Tuple, Optional, Union, List
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv, sqrtm

# Configure module-specific logger
logger = logging.getLogger(__name__)


class QuantizationError(Exception):
    """Custom exception for quantization sensitivity analysis errors."""
    pass


class QuantizationSensitivityAnalyzer:
    """
    Quantization Sensitivity Analysis of Symbolic Recurrence.
    
    This class implements the framework to assess how sensitive symbolic 
    recurrence metrics are to uncertainty in meta-state centroid definitions.
    It quantifies the contribution of quantization artifacts to observed 
    recurrence deficits, particularly important in AD where meta-state 
    boundaries are hypothesized to be blurred.
    
    The implementation follows the methodology described in the manuscript:
    - Perturbation of meta-state centroids within empirical variability bounds
    - Recomputation of symbolic sequences under perturbation
    - Variance-based sensitivity quantification
    - Relative sensitivity normalization
    
    Designed to work with group-level meta-states from all four datasets.
    """
    
    def __init__(
        self,
        perturbation_strength: float = 1.0,
        perturbation_count: int = 100,
        epsilon: float = 1e-6,
        random_state: int = 42
    ):
        """
        Initialize quantization sensitivity analyzer.
        
        Parameters
        ----------
        perturbation_strength : float, optional
            Strength of centroid perturbation relative to empirical variability (default: 1.0).
        perturbation_count : int, optional
            Number of perturbations to apply (default: 100).
        epsilon : float, optional
            Small constant for numerical stability (default: 1e-6).
        random_state : int, optional
            Random seed for reproducibility (default: 42).
        
        Raises
        ------
        QuantizationError
            If input parameters are invalid.
        """
        if perturbation_strength < 0:
            raise QuantizationError("perturbation_strength must be non-negative.")
        if perturbation_count <= 0:
            raise QuantizationError("perturbation_count must be positive.")
        if epsilon <= 0:
            raise QuantizationError("epsilon must be positive.")
        
        self.perturbation_strength = float(perturbation_strength)
        self.perturbation_count = int(perturbation_count)
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
    
    def _validate_inputs(
        self,
        C_traj: np.ndarray,
        meta_state_centroids: np.ndarray,
        meta_state_covariances: Optional[np.ndarray],
        symbolic_sequence: np.ndarray
    ) -> None:
        """
        Validate input data consistency.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        meta_state_centroids : np.ndarray
            Meta-state centroids of shape (n_states, m) where m = vectorized matrix dimension.
        meta_state_covariances : np.ndarray or None
            Meta-state covariances of shape (n_states, m, m) or None.
        symbolic_sequence : np.ndarray
            Original symbolic sequence of shape (n_times,).
        
        Raises
        ------
        QuantizationError
            If validation fails.
        """
        if not isinstance(C_traj, np.ndarray):
            raise QuantizationError("C_traj must be a NumPy array.")
        if C_traj.ndim != 3:
            raise QuantizationError(f"C_traj must be 3D, got {C_traj.ndim}D.")
        if not isinstance(meta_state_centroids, np.ndarray):
            raise QuantizationError("meta_state_centroids must be a NumPy array.")
        if meta_state_centroids.ndim != 2:
            raise QuantizationError(f"meta_state_centroids must be 2D, got {meta_state_centroids.ndim}D.")
        if not isinstance(symbolic_sequence, np.ndarray):
            raise QuantizationError("symbolic_sequence must be a NumPy array.")
        if symbolic_sequence.ndim != 1:
            raise QuantizationError(f"symbolic_sequence must be 1D, got {symbolic_sequence.ndim}D.")
        if C_traj.shape[0] != len(symbolic_sequence):
            raise QuantizationError(
                f"Time dimension mismatch: C_traj {C_traj.shape[0]} != symbolic_sequence {len(symbolic_sequence)}."
            )
        if not np.all(np.isfinite(C_traj)) or not np.all(np.isfinite(meta_state_centroids)):
            raise QuantizationError("Input data contains non-finite values.")
        if meta_state_covariances is not None:
            if meta_state_covariances.ndim != 3:
                raise QuantizationError(f"meta_state_covariances must be 3D, got {meta_state_covariances.ndim}D.")
            if meta_state_covariances.shape[0] != meta_state_centroids.shape[0]:
                raise QuantizationError("Covariance count mismatch with centroid count.")
    
    def _vectorize_iac_matrices(self,  np.ndarray) -> np.ndarray:
        """
        Vectorize IAC matrices by extracting lower triangular elements.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, d, d).
        
        Returns
        -------
        np.ndarray
            Vectorized trajectory of shape (n_times, m) where m = d(d+1)/2.
        """
        try:
            T, d, _ = C_traj.shape
            m = d * (d + 1) // 2
            X = np.zeros((T, m))
            
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        X[:, idx] = C_traj[:, i, j]
                    else:
                        # Average symmetric elements for numerical stability
                        X[:, idx] = (C_traj[:, i, j] + C_traj[:, j, i]) / 2.0
                    idx += 1
            
            return X
            
        except Exception as e:
            raise QuantizationError(f"IAC vectorization failed: {str(e)}")
    
    def _compute_empirical_covariances(
        self,
        X: np.ndarray,
        symbolic_sequence: np.ndarray,
        n_states: int
    ) -> np.ndarray:
        """
        Compute empirical covariances for each meta-state from assigned data points.
        
        Parameters
        ----------
        X : np.ndarray
            Vectorized IAC trajectory of shape (n_times, m).
        symbolic_sequence : np.ndarray
            Symbolic sequence of shape (n_times,).
        n_states : int
            Number of meta-states.
        
        Returns
        -------
        np.ndarray
            Empirical covariances of shape (n_states, m, m).
        """
        try:
            T, m = X.shape
            covariances = np.zeros((n_states, m, m))
            
            for k in range(n_states):
                # Find all time points assigned to meta-state k
                mask = symbolic_sequence == k
                if np.sum(mask) == 0:
                    # No data points assigned - use identity matrix scaled by global variance
                    global_var = np.var(X)
                    covariances[k] = np.eye(m) * global_var
                elif np.sum(mask) == 1:
                    # Only one data point - use small identity matrix
                    covariances[k] = np.eye(m) * self.epsilon
                else:
                    # Compute empirical covariance
                    X_k = X[mask]
                    cov_k = np.cov(X_k, rowvar=False)
                    # Ensure positive definiteness
                    cov_k = (cov_k + cov_k.T) / 2.0  # Symmetrize
                    eigenvals = np.linalg.eigvalsh(cov_k)
                    if np.min(eigenvals) < self.epsilon:
                        cov_k += np.eye(m) * (self.epsilon - np.min(eigenvals))
                    covariances[k] = cov_k
            
            return covariances
            
        except Exception as e:
            raise QuantizationError(f"Empirical covariance computation failed: {str(e)}")
    
    def _perturb_meta_state_centroids(
        self,
        centroids: np.ndarray,
        covariances: np.ndarray
    ) -> np.ndarray:
        """
        Perturb meta-state centroids within empirical variability bounds.
        
        Parameters
        ----------
        centroids : np.ndarray
            Original centroids of shape (n_states, m).
        covariances : np.ndarray
            Covariance matrices of shape (n_states, m, m).
        
        Returns
        -------
        np.ndarray
            Perturbed centroids of same shape.
        """
        try:
            n_states, m = centroids.shape
            perturbed_centroids = centroids.copy()
            
            for k in range(n_states):
                # Generate random perturbation from multivariate normal distribution
                cov_k = covariances[k]
                
                # Ensure covariance is positive definite
                try:
                    # Try Cholesky decomposition first (faster)
                    L = np.linalg.cholesky(cov_k)
                    random_vec = np.random.randn(m)
                    perturbation = self.perturbation_strength * L @ random_vec
                except np.linalg.LinAlgError:
                    # Fallback to eigenvalue decomposition
                    eigenvals, eigenvecs = np.linalg.eigh(cov_k)
                    eigenvals = np.maximum(eigenvals, self.epsilon)
                    sqrt_cov = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
                    random_vec = np.random.randn(m)
                    perturbation = self.perturbation_strength * sqrt_cov @ random_vec
                
                perturbed_centroids[k] += perturbation
            
            return perturbed_centroids
            
        except Exception as e:
            raise QuantizationError(f"Centroid perturbation failed: {str(e)}")
    
    def _assign_symbolic_sequence(
        self,
        X: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Assign symbolic sequence using nearest neighbor assignment to centroids.
        
        Parameters
        ----------
        X : np.ndarray
            Vectorized IAC trajectory of shape (n_times, m).
        centroids : np.ndarray
            Meta-state centroids of shape (n_states, m).
        
        Returns
        -------
        np.ndarray
            Symbolic sequence of shape (n_times,).
        """
        try:
            T, m = X.shape
            n_states = centroids.shape[0]
            symbolic_seq = np.zeros(T, dtype=int)
            
            # Compute distances to all centroids for all time points
            # Using broadcasting for efficiency
            diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (T, n_states, m)
            distances = np.linalg.norm(diff, axis=2)  # (T, n_states)
            
            # Assign to nearest centroid
            symbolic_seq = np.argmin(distances, axis=1)
            
            return symbolic_seq
            
        except Exception as e:
            raise QuantizationError(f"Symbolic sequence assignment failed: {str(e)}")
    
    def _compute_cramers_v_recurrence(
        self,
        symbolic_sequence: np.ndarray,
        tau_star: int
    ) -> float:
        """
        Compute Cramér's V recurrence at dominant lag tau_star.
        
        Parameters
        ----------
        symbolic_sequence : np.ndarray
            Symbolic sequence of shape (n_times,).
        tau_star : int
            Dominant lag in samples.
        
        Returns
        -------
        float
            Cramér's V recurrence value.
        """
        try:
            T = len(symbolic_sequence)
            if tau_star >= T:
                return 0.0
            
            s_t = symbolic_sequence[tau_star:]
            s_t_minus_tau = symbolic_sequence[:-tau_star]
            
            # Create contingency table
            unique_states = np.unique(np.concatenate([s_t, s_t_minus_tau]))
            n_states = len(unique_states)
            
            if n_states <= 1:
                return 0.0
            
            # Map states to indices
            state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
            s_t_idx = np.array([state_to_idx[s] for s in s_t])
            s_t_minus_tau_idx = np.array([state_to_idx[s] for s in s_t_minus_tau])
            
            # Build contingency table
            contingency = np.zeros((n_states, n_states), dtype=int)
            for i, j in zip(s_t_idx, s_t_minus_tau_idx):
                contingency[i, j] += 1
            
            # Compute chi-squared statistic
            chi2_stat = 0.0
            n = np.sum(contingency)
            if n == 0:
                return 0.0
            
            row_sums = np.sum(contingency, axis=1)
            col_sums = np.sum(contingency, axis=0)
            
            for i in range(n_states):
                for j in range(n_states):
                    expected = row_sums[i] * col_sums[j] / n
                    if expected > 0:
                        chi2_stat += (contingency[i, j] - expected) ** 2 / expected
            
            # Compute Cramér's V
            min_dim = min(n_states - 1, n_states - 1)
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2_stat / (n * min_dim))
            return float(np.clip(cramers_v, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Cramér's V computation failed: {str(e)}. Returning 0.0.")
            return 0.0
    
    def analyze_sensitivity(
        self,
        C_traj: np.ndarray,
        meta_state_centroids: np.ndarray,
        tau_star: int,
        meta_state_covariances: Optional[np.ndarray] = None,
        original_symbolic_sequence: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, float]:
        """
        Analyze quantization sensitivity of symbolic recurrence.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        meta_state_centroids : np.ndarray
            Meta-state centroids of shape (n_states, m).
        tau_star : int
            Dominant lag in samples.
        meta_state_covariances : np.ndarray, optional
            Meta-state covariances of shape (n_states, m, m). If None, computed from data.
        original_symbolic_sequence : np.ndarray, optional
            Original symbolic sequence. If None, computed from centroids.
        
        Returns
        -------
        tuple
            (relative_sensitivity, perturbed_recurrence_values, original_recurrence)
            - relative_sensitivity : float - \( \widetilde{\mathcal{Q}}(\tau^*) \)
            - perturbed_recurrence_values : np.ndarray - \( \widetilde{\rho}_S^{(r)}(\tau^*) \) for each perturbation
            - original_recurrence : float - \( \rho_S(\tau^*) \) for original centroids
        
        Raises
        ------
        QuantizationError
            If analysis fails.
        """
        try:
            logger.info("Analyzing quantization sensitivity...")
            
            # Vectorize IAC trajectory
            X = self._vectorize_iac_matrices(C_traj)
            
            # Compute original symbolic sequence if not provided
            if original_symbolic_sequence is None:
                original_symbolic_sequence = self._assign_symbolic_sequence(X, meta_state_centroids)
            
            # Compute original recurrence
            original_recurrence = self._compute_cramers_v_recurrence(original_symbolic_sequence, tau_star)
            
            # Compute empirical covariances if not provided
            if meta_state_covariances is None:
                meta_state_covariances = self._compute_empirical_covariances(
                    X, original_symbolic_sequence, meta_state_centroids.shape[0]
                )
            
            # Validate inputs
            self._validate_inputs(C_traj, meta_state_centroids, meta_state_covariances, original_symbolic_sequence)
            
            # Initialize perturbed recurrence values
            perturbed_recurrence_values = np.zeros(self.perturbation_count)
            
            # Apply perturbations and recompute recurrence
            for r in range(self.perturbation_count):
                # Perturb centroids
                perturbed_centroids = self._perturb_meta_state_centroids(
                    meta_state_centroids, meta_state_covariances
                )
                
                # Assign new symbolic sequence
                perturbed_symbolic_sequence = self._assign_symbolic_sequence(X, perturbed_centroids)
                
                # Compute recurrence for perturbed sequence
                perturbed_recurrence = self._compute_cramers_v_recurrence(
                    perturbed_symbolic_sequence, tau_star
                )
                perturbed_recurrence_values[r] = perturbed_recurrence
            
            # Compute sensitivity metrics
            sensitivity_variance = np.var(perturbed_recurrence_values)
            mean_perturbed_recurrence = np.mean(perturbed_recurrence_values)
            relative_sensitivity = sensitivity_variance / (mean_perturbed_recurrence ** 2 + self.epsilon)
            
            logger.info(f"Quantization sensitivity analysis completed. Relative sensitivity: {relative_sensitivity:.3f}")
            return float(relative_sensitivity), perturbed_recurrence_values, float(original_recurrence)
            
        except Exception as e:
            if isinstance(e, QuantizationError):
                raise
            raise QuantizationError(f"Quantization sensitivity analysis failed: {str(e)}")
    
    def compute_meta_state_voronoi_boundaries(
        self,
        centroids: np.ndarray
    ) -> dict:
        """
        Compute Voronoi cell boundaries for meta-state analysis (for visualization/debugging).
        
        Parameters
        ----------
        centroids : np.ndarray
            Meta-state centroids of shape (n_states, m).
        
        Returns
        -------
        dict
            Dictionary containing Voronoi boundary information.
        """
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(centroids)
            return {
                'vertices': vor.vertices,
                'ridge_vertices': vor.ridge_vertices,
                'ridge_points': vor.ridge_points,
                'point_region': vor.point_region,
                'regions': vor.regions
            }
        except Exception as e:
            logger.warning(f"Voronoi computation failed: {str(e)}. Returning empty dict.")
            return {}
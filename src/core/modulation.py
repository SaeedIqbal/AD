import numpy as np
import logging
from typing import Tuple, Optional, Union, List
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform

# Configure module-specific logger
logger = logging.getLogger(__name__)


class ModulationError(Exception):
    """Custom exception for modulation detection errors."""
    pass


class ExternalModulationDetector:
    """
    Detection of External Modulation in Meta-State Dynamics.
    
    This class implements the sliding-window conditional independence testing
    framework to detect nonstationary modulation of brain dynamics by 
    time-varying physiological drivers (e.g., HRV, respiration).
    
    The implementation follows the methodology described in the manuscript:
    - Vectorization of IAC matrices
    - Sliding-window conditional independence testing
    - Kernel-based or permutation-based statistical inference
    - Global modulation index computation
    
    Designed to work with auxiliary signals from BioFIND and PREVENT-AD,
    while gracefully handling missing signals in OpenNeuro and Mendeley.
    """
    
    def __init__(
        self,
        window_length: float = 5.0,
        stride: float = 1.0,
        autoregressive_order: int = 5,
        significance_alpha: float = 0.05,
        permutation_count: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize external modulation detector.
        
        Parameters
        ----------
        window_length : float, optional
            Length of sliding window in seconds (default: 5.0).
        stride : float, optional
            Stride between windows in seconds (default: 1.0).
        autoregressive_order : int, optional
            Order of autoregressive model for brain history (default: 5).
        significance_alpha : float, optional
            Significance level for hypothesis testing (default: 0.05).
        permutation_count : int, optional
            Number of permutations for null distribution (default: 1000).
        random_state : int, optional
            Random seed for reproducibility (default: 42).
        
        Raises
        ------
        ModulationError
            If input parameters are invalid.
        """
        if window_length <= 0:
            raise ModulationError("window_length must be positive.")
        if stride <= 0:
            raise ModulationError("stride must be positive.")
        if autoregressive_order <= 0:
            raise ModulationError("autoregressive_order must be positive.")
        if not (0 < significance_alpha < 1):
            raise ModulationError("significance_alpha must be in (0, 1).")
        if permutation_count <= 0:
            raise ModulationError("permutation_count must be positive.")
        
        self.window_length = float(window_length)
        self.stride = float(stride)
        self.autoregressive_order = int(autoregressive_order)
        self.significance_alpha = float(significance_alpha)
        self.permutation_count = int(permutation_count)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
    
    def _validate_inputs(
        self, 
        C_traj: np.ndarray, 
        u_t: np.ndarray, 
        sfreq: float
    ) -> None:
        """
        Validate input data consistency.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        u_t : np.ndarray
            Auxiliary signal of shape (n_times,).
        sfreq : float
            Sampling frequency in Hz.
        
        Raises
        ------
        ModulationError
            If validation fails.
        """
        if not isinstance(C_traj, np.ndarray) or not isinstance(u_t, np.ndarray):
            raise ModulationError("Inputs must be NumPy arrays.")
        if C_traj.ndim != 3:
            raise ModulationError(f"C_traj must be 3D, got {C_traj.ndim}D.")
        if u_t.ndim != 1:
            raise ModulationError(f"u_t must be 1D, got {u_t.ndim}D.")
        if C_traj.shape[0] != len(u_t):
            raise ModulationError(
                f"Time dimension mismatch: C_traj {C_traj.shape[0]} != u_t {len(u_t)}."
            )
        if sfreq <= 0:
            raise ModulationError("Sampling frequency must be positive.")
        if not np.all(np.isfinite(C_traj)) or not np.all(np.isfinite(u_t)):
            raise ModulationError("Input data contains non-finite values.")
    
    def _vectorize_iac_trajectory(self,  np.ndarray) -> np.ndarray:
        """
        Vectorize IAC trajectory by extracting lower triangular elements.
        
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
            raise ModulationError(f"IAC vectorization failed: {str(e)}")
    
    def _create_predictor_matrix(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Create autoregressive predictor matrix from vectorized IAC trajectory.
        
        Parameters
        ----------
        X : np.ndarray
            Vectorized IAC trajectory of shape (n_times, m).
        
        Returns
        -------
        np.ndarray
            Predictor matrix of shape (n_times, p*m) where p = autoregressive_order.
        """
        try:
            T, m = X.shape
            p = self.autoregressive_order
            
            if T <= p:
                raise ModulationError(f"Insufficient data: {T} samples <= {p} AR order.")
            
            # Initialize predictor matrix
            Z = np.zeros((T, p * m))
            
            # Fill predictor matrix with lagged values
            for t in range(p, T):
                for lag in range(p):
                    start_col = lag * m
                    end_col = (lag + 1) * m
                    Z[t, start_col:end_col] = X[t - lag - 1, :]
            
            # Remove initial rows that cannot be predicted
            Z = Z[p:, :]
            
            return Z
            
        except Exception as e:
            raise ModulationError(f"Predictor matrix creation failed: {str(e)}")
    
    def _conditional_permutation_test(
        self,
        x_t: np.ndarray,
        z_t: np.ndarray,
        u_t: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform conditional permutation test for independence.
        
        Parameters
        ----------
        x_t : np.ndarray
            Current brain state of shape (n_samples, m).
        z_t : np.ndarray
            Brain history predictor of shape (n_samples, p*m).
        u_t : np.ndarray
            Auxiliary signal of shape (n_samples,).
        
        Returns
        -------
        tuple
            (test_statistic, p_value)
        """
        try:
            n_samples = len(x_t)
            if n_samples < 10:
                return 0.0, 1.0
            
            # Compute test statistic (partial correlation)
            # Residualize x_t and u_t against z_t
            from scipy.linalg import lstsq
            
            # Residualize x_t against z_t
            coef_x, _, _, _ = lstsq(z_t, x_t, cond=None)
            x_residual = x_t - z_t @ coef_x
            
            # Residualize u_t against z_t
            coef_u, _, _, _ = lstsq(z_t, u_t.reshape(-1, 1), cond=None)
            u_residual = u_t - (z_t @ coef_u).flatten()
            
            # Compute partial correlation
            numerator = np.sum(x_residual * u_residual)
            denominator = np.sqrt(np.sum(x_residual**2) * np.sum(u_residual**2))
            
            if denominator == 0:
                partial_corr = 0.0
            else:
                partial_corr = numerator / denominator
            
            test_statistic = partial_corr**2 * (n_samples - self.autoregressive_order - 2) / (1 - partial_corr**2 + 1e-12)
            
            # Permutation test
            permuted_stats = []
            u_shuffled = u_t.copy()
            
            for _ in range(self.permutation_count):
                # Shuffle u_t while preserving its marginal distribution
                np.random.shuffle(u_shuffled)
                
                # Residualize shuffled u_t against z_t
                coef_u_perm, _, _, _ = lstsq(z_t, u_shuffled.reshape(-1, 1), cond=None)
                u_residual_perm = u_shuffled - (z_t @ coef_u_perm).flatten()
                
                # Compute partial correlation for permuted data
                numerator_perm = np.sum(x_residual * u_residual_perm)
                denominator_perm = np.sqrt(np.sum(x_residual**2) * np.sum(u_residual_perm**2))
                
                if denominator_perm == 0:
                    partial_corr_perm = 0.0
                else:
                    partial_corr_perm = numerator_perm / denominator_perm
                
                test_stat_perm = partial_corr_perm**2 * (n_samples - self.autoregressive_order - 2) / (1 - partial_corr_perm**2 + 1e-12)
                permuted_stats.append(test_stat_perm)
            
            # Compute p-value
            permuted_stats = np.array(permuted_stats)
            p_value = np.mean(permuted_stats >= test_statistic)
            
            return float(test_statistic), float(p_value)
            
        except Exception as e:
            logger.warning(f"Permutation test failed: {str(e)}. Returning non-significant result.")
            return 0.0, 1.0
    
    def _compute_kernel_conditional_independence(
        self,
        x_t: np.ndarray,
        z_t: np.ndarray,
        u_t: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute kernel-based conditional independence test (alternative method).
        
        Parameters
        ----------
        x_t : np.ndarray
            Current brain state of shape (n_samples, m).
        z_t : np.ndarray
            Brain history predictor of shape (n_samples, p*m).
        u_t : np.ndarray
            Auxiliary signal of shape (n_samples,).
        
        Returns
        -------
        tuple
            (test_statistic, p_value)
        """
        try:
            # This is a simplified Gaussian kernel implementation
            # For production use, consider more robust kernel methods
            
            def gaussian_kernel(X, Y=None, sigma=None):
                if Y is None:
                    Y = X
                if sigma is None:
                    sigma = np.median(pdist(X)) / 2 if len(X) > 1 else 1.0
                distances = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
                return np.exp(-distances / (2 * sigma ** 2))
            
            # Compute kernels
            K_x = gaussian_kernel(x_t)
            K_u = gaussian_kernel(u_t.reshape(-1, 1))
            K_z = gaussian_kernel(z_t)
            
            # Center kernels
            n = K_x.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            K_x_centered = H @ K_x @ H
            K_u_centered = H @ K_u @ H
            K_z_centered = H @ K_z @ H
            
            # Compute Hilbert-Schmidt Independence Criterion (HSIC)
            hsic = np.trace(K_x_centered @ K_u_centered) / (n - 1) ** 2
            
            # Simple p-value approximation (in practice, use permutation)
            # This is a placeholder - real implementation would use proper null distribution
            test_statistic = hsic
            p_value = 1.0 if hsic < 0.01 else 0.5  # Simplified for demonstration
            
            return float(test_statistic), float(p_value)
            
        except Exception as e:
            logger.warning(f"Kernel test failed: {str(e)}. Falling back to permutation test.")
            return self._conditional_permutation_test(x_t, z_t, u_t)
    
    def detect_modulation(
        self,
        C_traj: np.ndarray,
        u_t: np.ndarray,
        sfreq: float
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect external modulation using sliding-window conditional independence testing.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        u_t : np.ndarray
            Auxiliary signal of shape (n_times,).
        sfreq : float
            Sampling frequency in Hz.
        
        Returns
        -------
        tuple
            (global_modulation_index, p_values, window_starts, window_ends)
            - global_modulation_index : float - Γ (fraction of significant windows)
            - p_values : np.ndarray - p-values for each window
            - window_starts : np.ndarray - start indices of windows
            - window_ends : np.ndarray - end indices of windows
        
        Raises
        ------
        ModulationError
            If detection fails.
        """
        try:
            logger.info("Detecting external modulation...")
            
            # Handle missing auxiliary signal
            if u_t is None or len(u_t) == 0:
                logger.warning("No auxiliary signal provided. Returning Γ = 0.0.")
                T = C_traj.shape[0]
                window_starts = np.array([0])
                window_ends = np.array([T])
                return 0.0, np.array([1.0]), window_starts, window_ends
            
            # Validate inputs
            self._validate_inputs(C_traj, u_t, sfreq)
            
            # Convert time parameters to samples
            window_samples = int(self.window_length * sfreq)
            stride_samples = int(self.stride * sfreq)
            
            if window_samples <= self.autoregressive_order:
                raise ModulationError(
                    f"Window length {window_samples} samples <= AR order {self.autoregressive_order}."
                )
            
            T = C_traj.shape[0]
            if T < window_samples:
                raise ModulationError(
                    f"Insufficient data: {T} samples < {window_samples} window length."
                )
            
            # Vectorize IAC trajectory
            X = self._vectorize_iac_trajectory(C_traj)
            
            # Initialize results
            p_values = []
            window_starts = []
            window_ends = []
            significant_count = 0
            total_windows = 0
            
            # Sliding window analysis
            start_idx = 0
            while start_idx + window_samples <= T:
                end_idx = start_idx + window_samples
                
                # Extract window data
                X_window = X[start_idx:end_idx]
                u_window = u_t[start_idx:end_idx]
                
                # Create predictor matrix (brain history)
                Z_window = self._create_predictor_matrix(X_window)
                
                # Adjust u_window to match predictor length
                u_adjusted = u_window[self.autoregressive_order:]
                X_adjusted = X_window[self.autoregressive_order:]
                
                if len(u_adjusted) < 10:  # Minimum samples for reliable testing
                    start_idx += stride_samples
                    continue
                
                # Perform conditional independence test
                test_stat, p_val = self._conditional_permutation_test(
                    X_adjusted, Z_window, u_adjusted
                )
                
                p_values.append(p_val)
                window_starts.append(start_idx)
                window_ends.append(end_idx)
                total_windows += 1
                
                if p_val < self.significance_alpha:
                    significant_count += 1
                
                start_idx += stride_samples
            
            if total_windows == 0:
                logger.warning("No valid windows found. Returning Γ = 0.0.")
                return 0.0, np.array([]), np.array([]), np.array([])
            
            # Compute global modulation index
            global_modulation_index = significant_count / total_windows
            
            logger.info(f"Modulation detection completed. Γ = {global_modulation_index:.3f}")
            return (
                float(global_modulation_index),
                np.array(p_values),
                np.array(window_starts),
                np.array(window_ends)
            )
            
        except Exception as e:
            if isinstance(e, ModulationError):
                raise
            raise ModulationError(f"Modulation detection failed: {str(e)}")
    
    def has_auxiliary_signal(self, dataset_name: str) -> bool:
        """
        Check if a dataset typically has auxiliary physiological signals.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        
        Returns
        -------
        bool
            True if auxiliary signals are typically available.
        """
        available_datasets = ['BioFIND', 'PREVENT-AD']
        return any(name in dataset_name for name in available_datasets)
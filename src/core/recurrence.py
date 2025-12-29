import numpy as np
import logging
from typing import Tuple, Optional, Union, List
from scipy.spatial.distance import cosine
from scipy.stats import zscore

# Configure module-specific logger
logger = logging.getLogger(__name__)


class RecurrenceError(Exception):
    """Custom exception for recurrence computation errors."""
    pass


class ContinuousRecurrence:
    """
    Continuous Recurrence Quantification on IAC Trajectories.
    
    This class implements the continuous recurrence framework described in 
    the methodology, which operates directly on the native geometry of 
    functional connectivity trajectories without symbolic discretization.
    
    Key features:
    - Kernelized autocorrelation using Frobenius inner product
    - AAFT surrogate-based normalization
    - Dominant lag detection
    - Robust to boundary ambiguity in meta-state space
    
    The implementation is optimized for the four datasets in the study:
    OpenNeuro ds004504, BioFIND, Mendeley Olfactory, and PREVENT-AD.
    """
    
    def __init__(
        self,
        tau_max: int = 100,
        surrogate_count: int = 100,
        random_state: int = 42
    ):
        """
        Initialize continuous recurrence computation.
        
        Parameters
        ----------
        tau_max : int, optional
            Maximum lag for recurrence analysis (default: 100).
        surrogate_count : int, optional
            Number of AAFT surrogates for normalization (default: 100).
        random_state : int, optional
            Random seed for surrogate generation (default: 42).
        
        Raises
        ------
        RecurrenceError
            If input parameters are invalid.
        """
        if tau_max <= 0:
            raise RecurrenceError("tau_max must be positive.")
        if surrogate_count <= 0:
            raise RecurrenceError("surrogate_count must be positive.")
        
        self.tau_max = int(tau_max)
        self.surrogate_count = int(surrogate_count)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
    
    def _validate_iac_trajectory(self,  np.ndarray) -> None:
        """
        Validate IAC trajectory input.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        
        Raises
        ------
        RecurrenceError
            If validation fails.
        """
        if not isinstance(C_traj, np.ndarray):
            raise RecurrenceError("IAC trajectory must be a NumPy array.")
        if C_traj.ndim != 3:
            raise RecurrenceError(f"IAC trajectory must be 3D, got {C_traj.ndim}D.")
        if C_traj.shape[0] == 0 or C_traj.shape[1] == 0 or C_traj.shape[2] == 0:
            raise RecurrenceError("IAC trajectory cannot be empty.")
        if C_traj.shape[1] != C_traj.shape[2]:
            raise RecurrenceError("IAC trajectory must contain square matrices.")
        if not np.all(np.isfinite(C_traj)):
            raise RecurrenceError("IAC trajectory contains non-finite values.")
        if C_traj.shape[0] < self.tau_max + 1:
            raise RecurrenceError(
                f"IAC trajectory too short: {C_traj.shape[0]} samples < {self.tau_max + 1} required."
            )
    
    def _compute_continuous_recurrence(self,  np.ndarray) -> np.ndarray:
        """
        Compute continuous recurrence function using kernelized autocorrelation.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, d, d).
        
        Returns
        -------
        np.ndarray
            Continuous recurrence function of shape (tau_max,).
        """
        try:
            T, d, _ = C_traj.shape
            
            # Vectorize symmetric matrices (lower triangular part)
            m = d * (d + 1) // 2
            X = np.zeros((T, m))
            
            # Extract lower triangular elements efficiently
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        X[:, idx] = C_traj[:, i, j]
                        idx += 1
                    else:
                        # Average symmetric elements for numerical stability
                        X[:, idx] = (C_traj[:, i, j] + C_traj[:, j, i]) / 2.0
                        idx += 1
            
            # Compute Frobenius norms
            norms = np.linalg.norm(X, axis=1)
            
            # Compute kernelized autocorrelation
            rho = np.zeros(self.tau_max)
            for tau in range(1, self.tau_max + 1):
                if T - tau <= 0:
                    rho[tau - 1] = 0.0
                    continue
                
                # Compute cosine similarity (equivalent to normalized Frobenius inner product)
                dot_products = np.sum(X[:-tau] * X[tau:], axis=1)
                denominators = norms[:-tau] * norms[tau:]
                
                # Handle zero denominators
                valid_mask = denominators > 1e-12
                if np.any(valid_mask):
                    similarities = dot_products[valid_mask] / denominators[valid_mask]
                    rho[tau - 1] = np.mean(similarities)
                else:
                    rho[tau - 1] = 0.0
            
            return rho
            
        except Exception as e:
            raise RecurrenceError(f"Continuous recurrence computation failed: {str(e)}")
    
    def _generate_aaft_surrogate(self,  np.ndarray) -> np.ndarray:
        """
        Generate Amplitude-Adjusted Fourier Transform (AAFT) surrogate.
        
        Parameters
        ----------
        original_signal : np.ndarray
            Original time series of shape (n_times,).
        
        Returns
        -------
        np.ndarray
            AAFT surrogate of same shape.
        """
        try:
            # Step 1: Gaussianize the original signal
            ranks = np.argsort(np.argsort(original_signal))
            gaussian_signal = np.random.randn(len(original_signal))
            gaussian_signal_sorted = np.sort(gaussian_signal)
            gaussianized = gaussian_signal_sorted[ranks]
            
            # Step 2: Randomize phases in Fourier domain
            fft_gauss = np.fft.rfft(gaussianized)
            phases = np.random.uniform(0, 2 * np.pi, len(fft_gauss))
            fft_surrogate = np.abs(fft_gauss) * np.exp(1j * phases)
            surrogate_gauss = np.fft.irfft(fft_surrogate, n=len(original_signal))
            
            # Step 3: De-Gaussianize to original amplitude distribution
            surrogate_ranks = np.argsort(np.argsort(surrogate_gauss))
            original_sorted = np.sort(original_signal)
            surrogate = original_sorted[surrogate_ranks]
            
            return surrogate
            
        except Exception as e:
            raise RecurrenceError(f"AAFT surrogate generation failed: {str(e)}")
    
    def _compute_surrogate_recurrence(self,  np.ndarray) -> np.ndarray:
        """
        Compute recurrence function for AAFT surrogates.
        
        Parameters
        ----------
        C_traj : np.ndarray
            Original IAC trajectory of shape (n_times, d, d).
        
        Returns
        -------
        np.ndarray
            Mean surrogate recurrence function of shape (tau_max,).
        """
        try:
            T, d, _ = C_traj.shape
            surrogate_rho_sum = np.zeros(self.tau_max)
            
            # Vectorize original trajectory
            m = d * (d + 1) // 2
            X_original = np.zeros((T, m))
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        X_original[:, idx] = C_traj[:, i, j]
                    else:
                        X_original[:, idx] = (C_traj[:, i, j] + C_traj[:, j, i]) / 2.0
                    idx += 1
            
            # Generate surrogates for each dimension
            for surrogate_idx in range(self.surrogate_count):
                X_surrogate = np.zeros_like(X_original)
                
                # Generate surrogate for each vectorized dimension
                for dim in range(m):
                    X_surrogate[:, dim] = self._generate_aaft_surrogate(X_original[:, dim])
                
                # Compute norms for surrogate
                norms_surrogate = np.linalg.norm(X_surrogate, axis=1)
                
                # Compute surrogate recurrence
                rho_surrogate = np.zeros(self.tau_max)
                for tau in range(1, self.tau_max + 1):
                    if T - tau <= 0:
                        continue
                    
                    dot_products = np.sum(X_surrogate[:-tau] * X_surrogate[tau:], axis=1)
                    denominators = norms_surrogate[:-tau] * norms_surrogate[tau:]
                    valid_mask = denominators > 1e-12
                    
                    if np.any(valid_mask):
                        similarities = dot_products[valid_mask] / denominators[valid_mask]
                        rho_surrogate[tau - 1] = np.mean(similarities)
                
                surrogate_rho_sum += rho_surrogate
            
            return surrogate_rho_sum / self.surrogate_count
            
        except Exception as e:
            raise RecurrenceError(f"Surrogate recurrence computation failed: {str(e)}")
    
    def compute_normalized_recurrence(
        self, 
        C_traj: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Compute normalized continuous recurrence function.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        
        Returns
        -------
        tuple
            (normalized_rho, tau_star, raw_rho) where:
            - normalized_rho : np.ndarray of shape (tau_max,) - normalized recurrence
            - tau_star : int - dominant lag (1-indexed)
            - raw_rho : np.ndarray of shape (tau_max,) - raw recurrence before normalization
        
        Raises
        ------
        RecurrenceError
            If computation fails.
        """
        try:
            logger.info("Computing normalized continuous recurrence...")
            
            # Validate input
            self._validate_iac_trajectory(C_traj)
            
            # Compute raw continuous recurrence
            raw_rho = self._compute_continuous_recurrence(C_traj)
            
            # Compute surrogate recurrence for normalization
            surrogate_rho = self._compute_surrogate_recurrence(C_traj)
            
            # Normalize (with epsilon to avoid division by zero)
            epsilon = 1e-12
            normalized_rho = raw_rho / (surrogate_rho + epsilon)
            
            # Identify dominant lag (1-indexed, corresponding to actual lag in samples)
            tau_star = np.argmax(normalized_rho) + 1
            
            logger.info(f"Dominant lag: {tau_star} samples")
            return normalized_rho, tau_star, raw_rho
            
        except Exception as e:
            if isinstance(e, RecurrenceError):
                raise
            raise RecurrenceError(f"Normalized recurrence computation failed: {str(e)}")
    
    def compute_dominant_lag_seconds(self, tau_star: int, sfreq: float) -> float:
        """
        Convert dominant lag from samples to seconds.
        
        Parameters
        ----------
        tau_star : int
            Dominant lag in samples.
        sfreq : float
            Sampling frequency in Hz.
        
        Returns
        -------
        float
            Dominant lag in seconds.
        """
        if sfreq <= 0:
            raise RecurrenceError("Sampling frequency must be positive.")
        return float(tau_star) / float(sfreq)


class SymbolicRecurrence:
    """
    Symbolic Recurrence Computation (for comparison and quantization sensitivity analysis).
    
    This class computes traditional symbolic recurrence using Cramér's V,
    as used in the reference study, to enable direct comparison with 
    continuous recurrence and quantization sensitivity analysis.
    """
    
    def __init__(self, tau_max: int = 100):
        """
        Initialize symbolic recurrence computation.
        
        Parameters
        ----------
        tau_max : int, optional
            Maximum lag for recurrence analysis (default: 100).
        """
        if tau_max <= 0:
            raise RecurrenceError("tau_max must be positive.")
        self.tau_max = int(tau_max)
    
    def _validate_symbolic_sequence(self,  np.ndarray) -> None:
        """Validate symbolic sequence input."""
        if not isinstance(s, np.ndarray):
            raise RecurrenceError("Symbolic sequence must be a NumPy array.")
        if s.ndim != 1:
            raise RecurrenceError("Symbolic sequence must be 1D.")
        if len(s) == 0:
            raise RecurrenceError("Symbolic sequence cannot be empty.")
        if not np.issubdtype(s.dtype, np.integer):
            raise RecurrenceError("Symbolic sequence must contain integers.")
        if np.min(s) < 0:
            raise RecurrenceError("Symbolic sequence cannot contain negative values.")
    
    def _cramers_v(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Cramér's V statistic for two categorical variables.
        
        Parameters
        ----------
        x, y : np.ndarray
            Categorical arrays of same length.
        
        Returns
        -------
        float
            Cramér's V statistic in [0, 1].
        """
        try:
            # Create contingency table
            x_unique, x_indices = np.unique(x, return_inverse=True)
            y_unique, y_indices = np.unique(y, return_inverse=True)
            
            contingency = np.zeros((len(x_unique), len(y_unique)), dtype=int)
            for i, j in zip(x_indices, y_indices):
                contingency[i, j] += 1
            
            # Compute chi-squared statistic
            chi2_stat = 0.0
            n = np.sum(contingency)
            if n == 0:
                return 0.0
            
            row_sums = np.sum(contingency, axis=1)
            col_sums = np.sum(contingency, axis=0)
            
            for i in range(contingency.shape[0]):
                for j in range(contingency.shape[1]):
                    expected = row_sums[i] * col_sums[j] / n
                    if expected > 0:
                        chi2_stat += (contingency[i, j] - expected) ** 2 / expected
            
            # Compute Cramér's V
            min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2_stat / (n * min_dim))
            return float(np.clip(cramers_v, 0.0, 1.0))
            
        except Exception as e:
            raise RecurrenceError(f"Cramér's V computation failed: {str(e)}")
    
    def compute_symbolic_recurrence(self,  np.ndarray) -> np.ndarray:
        """
        Compute symbolic recurrence using Cramér's V.
        
        Parameters
        ----------
        s : np.ndarray
            Symbolic sequence of shape (n_times,).
        
        Returns
        -------
        np.ndarray
            Symbolic recurrence function of shape (tau_max,).
        """
        try:
            self._validate_symbolic_sequence(s)
            T = len(s)
            rho_s = np.zeros(self.tau_max)
            
            for tau in range(1, self.tau_max + 1):
                if T - tau <= 0:
                    rho_s[tau - 1] = 0.0
                    continue
                rho_s[tau - 1] = self._cramers_v(s[:-tau], s[tau:])
            
            return rho_s
            
        except Exception as e:
            if isinstance(e, RecurrenceError):
                raise
            raise RecurrenceError(f"Symbolic recurrence computation failed: {str(e)}")
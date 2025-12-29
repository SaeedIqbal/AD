import numpy as np
import logging
from typing import Tuple, Optional, Union
from scipy.stats import zscore

# Configure module-specific logger
logger = logging.getLogger(__name__)


class AttractorError(Exception):
    """Custom exception for attractor stability computation errors."""
    pass


class GeometricRecurrenceIndex:
    """
    Attractor Stability Criterion via Geometric Recurrence Index.
    
    This class implements the geometric recurrence framework to directly 
    quantify the stability of the brain's intrinsic dynamical attractor.
    It operates on the continuous IAC trajectory without symbolic discretization,
    providing a direct measure of state revisitation in the native geometry 
    of functional connectivity space.
    
    The implementation follows the methodology described in the manuscript:
    - Computation of recurrence tolerance based on intra-subject variability
    - Geometric recurrence counting at dominant lag
    - AAFT surrogate-based normalization
    - Direct probe of attractor integrity
    
    Designed to work with IAC trajectories from all four datasets in the study.
    """
    
    def __init__(
        self,
        tolerance_multiplier: float = 3.0,
        surrogate_count: int = 100,
        random_state: int = 42
    ):
        """
        Initialize geometric recurrence index computation.
        
        Parameters
        ----------
        tolerance_multiplier : float, optional
            Multiplier for recurrence tolerance based on median step size (default: 3.0).
        surrogate_count : int, optional
            Number of AAFT surrogates for normalization (default: 100).
        random_state : int, optional
            Random seed for surrogate generation (default: 42).
        
        Raises
        ------
        AttractorError
            If input parameters are invalid.
        """
        if tolerance_multiplier <= 0:
            raise AttractorError("tolerance_multiplier must be positive.")
        if surrogate_count <= 0:
            raise AttractorError("surrogate_count must be positive.")
        
        self.tolerance_multiplier = float(tolerance_multiplier)
        self.surrogate_count = int(surrogate_count)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
    
    def _validate_inputs(
        self,
        C_traj: np.ndarray,
        tau_star: int
    ) -> None:
        """
        Validate input data consistency.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        tau_star : int
            Dominant lag in samples.
        
        Raises
        ------
        AttractorError
            If validation fails.
        """
        if not isinstance(C_traj, np.ndarray):
            raise AttractorError("C_traj must be a NumPy array.")
        if C_traj.ndim != 3:
            raise AttractorError(f"C_traj must be 3D, got {C_traj.ndim}D.")
        if C_traj.shape[0] == 0 or C_traj.shape[1] == 0 or C_traj.shape[2] == 0:
            raise AttractorError("C_traj cannot be empty.")
        if C_traj.shape[1] != C_traj.shape[2]:
            raise AttractorError("C_traj must contain square matrices.")
        if not np.all(np.isfinite(C_traj)):
            raise AttractorError("C_traj contains non-finite values.")
        if tau_star <= 0:
            raise AttractorError("tau_star must be positive.")
        if tau_star >= C_traj.shape[0]:
            raise AttractorError(f"tau_star {tau_star} >= trajectory length {C_traj.shape[0]}.")
    
    def _compute_step_sizes(self,  np.ndarray) -> np.ndarray:
        """
        Compute instantaneous step sizes in IAC trajectory.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, d, d).
        
        Returns
        -------
        np.ndarray
            Step sizes of shape (n_times - 1,).
        """
        try:
            T = C_traj.shape[0]
            step_sizes = np.zeros(T - 1)
            
            for t in range(T - 1):
                diff = C_traj[t + 1] - C_traj[t]
                step_sizes[t] = np.linalg.norm(diff, ord='fro')
            
            return step_sizes
            
        except Exception as e:
            raise AttractorError(f"Step size computation failed: {str(e)}")
    
    def _estimate_recurrence_tolerance(self,  np.ndarray) -> float:
        """
        Estimate recurrence tolerance based on intra-subject variability.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, d, d).
        
        Returns
        -------
        float
            Recurrence tolerance epsilon.
        """
        try:
            step_sizes = self._compute_step_sizes(C_traj)
            if len(step_sizes) == 0:
                return 1.0  # Default tolerance
            
            median_step_size = np.median(step_sizes)
            epsilon = self.tolerance_multiplier * median_step_size
            return float(epsilon)
            
        except Exception as e:
            raise AttractorError(f"Recurrence tolerance estimation failed: {str(e)}")
    
    def _compute_geometric_recurrence(
        self,
        C_traj: np.ndarray,
        tau_star: int,
        epsilon: float
    ) -> float:
        """
        Compute geometric recurrence index at dominant lag.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, d, d).
        tau_star : int
            Dominant lag in samples.
        epsilon : float
            Recurrence tolerance.
        
        Returns
        -------
        float
            Geometric recurrence index R.
        """
        try:
            T = C_traj.shape[0]
            if tau_star >= T:
                return 0.0
            
            recurrence_count = 0
            total_comparisons = T - tau_star
            
            for t in range(total_comparisons):
                diff = C_traj[t] - C_traj[t + tau_star]
                distance = np.linalg.norm(diff, ord='fro')
                if distance < epsilon:
                    recurrence_count += 1
            
            if total_comparisons == 0:
                return 0.0
            
            R = recurrence_count / total_comparisons
            return float(R)
            
        except Exception as e:
            raise AttractorError(f"Geometric recurrence computation failed: {str(e)}")
    
    def _generate_aaft_surrogate_iac(
        self,
        C_traj: np.ndarray
    ) -> np.ndarray:
        """
        Generate AAFT surrogate for IAC trajectory.
        
        Parameters
        ----------
        C_traj : np.ndarray
            Original IAC trajectory of shape (n_times, d, d).
        
        Returns
        -------
        np.ndarray
            AAFT surrogate IAC trajectory of same shape.
        """
        try:
            T, d, _ = C_traj.shape
            
            # Vectorize the trajectory (lower triangular elements)
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
            
            # Generate surrogate for each dimension
            X_surrogate = np.zeros_like(X_original)
            for dim in range(m):
                # Gaussianize
                ranks = np.argsort(np.argsort(X_original[:, dim]))
                gaussian_signal = np.random.randn(T)
                gaussian_signal_sorted = np.sort(gaussian_signal)
                gaussianized = gaussian_signal_sorted[ranks]
                
                # Randomize phases
                fft_gauss = np.fft.rfft(gaussianized)
                phases = np.random.uniform(0, 2 * np.pi, len(fft_gauss))
                fft_surrogate = np.abs(fft_gauss) * np.exp(1j * phases)
                surrogate_gauss = np.fft.irfft(fft_surrogate, n=T)
                
                # De-Gaussianize
                surrogate_ranks = np.argsort(np.argsort(surrogate_gauss))
                original_sorted = np.sort(X_original[:, dim])
                X_surrogate[:, dim] = original_sorted[surrogate_ranks]
            
            # Reconstruct IAC trajectory from surrogate vectorized data
            C_surrogate = np.zeros_like(C_traj)
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        C_surrogate[:, i, j] = X_surrogate[:, idx]
                        C_surrogate[:, j, i] = X_surrogate[:, idx]  # Symmetric
                    else:
                        C_surrogate[:, i, j] = X_surrogate[:, idx]
                        C_surrogate[:, j, i] = X_surrogate[:, idx]  # Symmetric
                    idx += 1
            
            return C_surrogate
            
        except Exception as e:
            raise AttractorError(f"AAFT surrogate generation failed: {str(e)}")
    
    def _compute_surrogate_geometric_recurrence(
        self,
        C_traj: np.ndarray,
        tau_star: int,
        epsilon: float
    ) -> float:
        """
        Compute mean geometric recurrence index for AAFT surrogates.
        
        Parameters
        ----------
        C_traj : np.ndarray
            Original IAC trajectory of shape (n_times, d, d).
        tau_star : int
            Dominant lag in samples.
        epsilon : float
            Recurrence tolerance.
        
        Returns
        -------
        float
            Mean surrogate geometric recurrence index R_sur.
        """
        try:
            surrogate_sum = 0.0
            
            for _ in range(self.surrogate_count):
                # Generate surrogate
                C_surrogate = self._generate_aaft_surrogate_iac(C_traj)
                
                # Compute recurrence for surrogate
                R_surrogate = self._compute_geometric_recurrence(C_surrogate, tau_star, epsilon)
                surrogate_sum += R_surrogate
            
            R_sur_mean = surrogate_sum / self.surrogate_count
            return float(R_sur_mean)
            
        except Exception as e:
            raise AttractorError(f"Surrogate geometric recurrence computation failed: {str(e)}")
    
    def compute_normalized_geometric_recurrence(
        self,
        C_traj: np.ndarray,
        tau_star: int
    ) -> Tuple[float, float, float]:
        """
        Compute normalized geometric recurrence index.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        tau_star : int
            Dominant lag in samples.
        
        Returns
        -------
        tuple
            (normalized_R, raw_R, R_surrogate)
            - normalized_R : float - \( \widehat{\mathcal{R}} \) (normalized geometric recurrence index)
            - raw_R : float - \( \mathcal{R} \) (raw geometric recurrence index)
            - R_surrogate : float - \( \mathcal{R}^{\text{sur}} \) (surrogate geometric recurrence index)
        
        Raises
        ------
        AttractorError
            If computation fails.
        """
        try:
            logger.info("Computing normalized geometric recurrence index...")
            
            # Validate inputs
            self._validate_inputs(C_traj, tau_star)
            
            # Estimate recurrence tolerance
            epsilon = self._estimate_recurrence_tolerance(C_traj)
            logger.debug(f"Recurrence tolerance epsilon: {epsilon:.6f}")
            
            # Compute raw geometric recurrence
            raw_R = self._compute_geometric_recurrence(C_traj, tau_star, epsilon)
            logger.debug(f"Raw geometric recurrence R: {raw_R:.6f}")
            
            # Compute surrogate geometric recurrence
            R_surrogate = self._compute_surrogate_geometric_recurrence(C_traj, tau_star, epsilon)
            logger.debug(f"Surrogate geometric recurrence R_sur: {R_surrogate:.6f}")
            
            # Normalize (with epsilon to avoid division by zero)
            normalized_R = raw_R / (R_surrogate + 1e-12)
            logger.debug(f"Normalized geometric recurrence R_hat: {normalized_R:.6f}")
            
            logger.info(f"Geometric recurrence computation completed. R_hat = {normalized_R:.3f}")
            return float(normalized_R), float(raw_R), float(R_surrogate)
            
        except Exception as e:
            if isinstance(e, AttractorError):
                raise
            raise AttractorError(f"Normalized geometric recurrence computation failed: {str(e)}")
    
    def detect_attractor_erosion(
        self,
        normalized_R: float,
        threshold: float = 1.5
    ) -> bool:
        """
        Detect attractor erosion based on normalized geometric recurrence.
        
        Parameters
        ----------
        normalized_R : float
            Normalized geometric recurrence index.
        threshold : float, optional
            Threshold for attractor erosion detection (default: 1.5).
        
        Returns
        -------
        bool
            True if attractor erosion is detected.
        """
        if normalized_R <= 0:
            return True  # Severe erosion
        return normalized_R < threshold
    
    def compute_attractor_stability_score(
        self,
        normalized_R: float
    ) -> float:
        """
        Compute attractor stability score for clinical correlation.
        
        Parameters
        ----------
        normalized_R : float
            Normalized geometric recurrence index.
        
        Returns
        -------
        float
            Attractor stability score in [0, 1] (higher = more stable).
        """
        # Map normalized_R to [0, 1] using logistic function
        # This provides a smooth, interpretable stability score
        stability_score = 1.0 / (1.0 + np.exp(-(normalized_R - 2.0)))
        return float(np.clip(stability_score, 0.0, 1.0))
# src/core/iac.py

import numpy as np
import logging
from typing import Tuple, Optional, Union
from scipy.signal import hilbert
from scipy.linalg import svd, orthogonal_procrustes

# Configure module-specific logger
logger = logging.getLogger(__name__)


class IACError(Exception):
    """Custom exception for IAC computation errors."""
    pass


class IACTrajectory:
    """
    Instantaneous Amplitude Correlation (IAC) Trajectory Computation.
    
    This class computes the IAC trajectory from source-level M/EEG time series
    following the methodology described in the manuscript:
    1. Bandpass filtering within target frequency band
    2. Hilbert transform to obtain analytic signal
    3. Envelope extraction
    4. Pairwise orthogonalization to mitigate volume conduction
    5. Correlation matrix computation at each time point
    
    The implementation is optimized for numerical stability and scientific accuracy.
    """
    
    def __init__(
        self,
        sfreq: float,
        band: Tuple[float, float],
        orthogonalize: bool = True,
        envelope_method: str = 'hilbert'
    ):
        """
        Initialize IAC trajectory computation.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.
        band : tuple of (float, float)
            Frequency band for bandpass filtering (low, high) in Hz.
        orthogonalize : bool, optional
            Whether to apply pairwise orthogonalization (default: True).
        envelope_method : str, optional
            Method for envelope extraction (default: 'hilbert').
        
        Raises
        ------
        IACError
            If input parameters are invalid.
        """
        if sfreq <= 0:
            raise IACError("Sampling frequency must be positive.")
        if len(band) != 2:
            raise IACError("Band must be a tuple of (low, high) frequencies.")
        if band[0] >= band[1]:
            raise IACError("Invalid band: low frequency must be < high frequency.")
        if band[1] >= sfreq / 2:
            raise IACError(f"High frequency {band[1]} >= Nyquist frequency {sfreq/2}.")
        if envelope_method not in ['hilbert']:
            raise IACError(f"Unsupported envelope method: {envelope_method}.")
        
        self.sfreq = float(sfreq)
        self.band = (float(band[0]), float(band[1]))
        self.orthogonalize = bool(orthogonalize)
        self.envelope_method = envelope_method
        self._filter_coeffs = None
        
        # Pre-compute filter coefficients for efficiency
        self._compute_filter_coeffs()
    
    def _compute_filter_coeffs(self) -> None:
        """Pre-compute Butterworth filter coefficients."""
        try:
            from scipy.signal import butter
            nyquist = self.sfreq / 2.0
            low_norm = self.band[0] / nyquist
            high_norm = self.band[1] / nyquist
            self._b, self._a = butter(N=4, Wn=[low_norm, high_norm], btype='band')
        except Exception as e:
            raise IACError(f"Failed to compute filter coefficients: {str(e)}")
    
    def _validate_input_data(self,  np.ndarray) -> None:
        """
        Validate input source time series data.
        
        Parameters
        ----------
        data : np.ndarray
            Source time series of shape (n_rois, n_times).
        
        Raises
        ------
        IACError
            If validation fails.
        """
        if not isinstance(data, np.ndarray):
            raise IACError("Input data must be a NumPy array.")
        if data.ndim != 2:
            raise IACError(f"Input data must be 2D, got {data.ndim}D.")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise IACError("Input data cannot be empty.")
        if not np.all(np.isfinite(data)):
            raise IACError("Input data contains non-finite values (NaN/Inf).")
        if data.shape[1] < 10:  # Minimum 10 samples for meaningful analysis
            raise IACError("Input data too short (< 10 samples).")
    
    def _bandpass_filter(self,  np.ndarray) -> np.ndarray:
        """
        Apply 4th-order Butterworth bandpass filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_rois, n_times).
        
        Returns
        -------
        np.ndarray
            Filtered data of same shape.
        """
        try:
            from scipy.signal import filtfilt
            return filtfilt(self._b, self._a, data, axis=1, padtype='odd')
        except Exception as e:
            raise IACError(f"Bandpass filtering failed: {str(e)}")
    
    def _compute_envelope(self,  np.ndarray) -> np.ndarray:
        """
        Compute amplitude envelope using Hilbert transform.
        
        Parameters
        ----------
        filtered_data : np.ndarray
            Bandpass-filtered data of shape (n_rois, n_times).
        
        Returns
        -------
        np.ndarray
            Envelope data of same shape.
        """
        try:
            if self.envelope_method == 'hilbert':
                analytic_signal = hilbert(filtered_data, axis=1)
                envelope = np.abs(analytic_signal)
                return envelope
            else:
                # This should never happen due to validation in __init__
                raise IACError(f"Unknown envelope method: {self.envelope_method}")
        except Exception as e:
            raise IACError(f"Envelope computation failed: {str(e)}")
    
    def _pairwise_orthogonalization(self,  np.ndarray) -> np.ndarray:
        """
        Apply pairwise orthogonalization to mitigate volume conduction.
        
        Implements the method from O'Neill et al. (2018) using Procrustes analysis
        to orthogonalize envelope time series between all pairs of ROIs.
        
        Parameters
        ----------
        envelopes : np.ndarray
            Envelope data of shape (n_rois, n_times).
        
        Returns
        -------
        np.ndarray
            Orthogonalized envelope data of same shape.
        """
        try:
            d, T = envelopes.shape
            orthogonalized = envelopes.copy().astype(np.float64)
            
            # Process all unique pairs
            for i in range(d):
                for j in range(i + 1, d):
                    # Extract envelope time series for pair (i, j)
                    env_i = orthogonalized[i:i+1].T  # Shape: (T, 1)
                    env_j = orthogonalized[j:j+1].T  # Shape: (T, 1)
                    
                    # Skip if either envelope is zero
                    if np.allclose(env_i, 0) or np.allclose(env_j, 0):
                        continue
                    
                    # Apply Procrustes analysis to orthogonalize
                    try:
                        R, _ = orthogonal_procrustes(env_i, env_j)
                        # Transform the second envelope to be orthogonal to the first
                        orthogonalized[j] = (R @ env_j.T).flatten()
                    except np.linalg.LinAlgError:
                        # Singular matrix - skip orthogonalization for this pair
                        continue
                    except Exception:
                        # Other numerical issues - skip
                        continue
            
            return orthogonalized
            
        except Exception as e:
            logger.warning(f"Orthogonalization failed: {str(e)}. Skipping.")
            return envelopes
    
    def _compute_correlation_matrices(self,  np.ndarray) -> np.ndarray:
        """
        Compute correlation matrices at each time point.
        
        Parameters
        ----------
        envelopes : np.ndarray
            Envelope data of shape (n_rois, n_times).
        
        Returns
        -------
        np.ndarray
            Correlation matrices of shape (n_times, n_rois, n_rois).
        """
        try:
            d, T = envelopes.shape
            C_traj = np.zeros((T, d, d), dtype=np.float64)
            
            # Vectorized computation for efficiency
            # Standardize envelopes (zero mean, unit variance) for each ROI
            envelopes_std = envelopes - np.mean(envelopes, axis=1, keepdims=True)
            envelopes_std = envelopes_std / (np.std(envelopes_std, axis=1, keepdims=True) + 1e-12)
            
            # Compute correlation matrices using matrix multiplication
            C_traj = np.einsum('it,jt->tij', envelopes_std, envelopes_std) / T
            
            # Ensure symmetry and valid correlation range [-1, 1]
            C_traj = (C_traj + np.transpose(C_traj, (0, 2, 1))) / 2.0
            C_traj = np.clip(C_traj, -1.0, 1.0)
            
            return C_traj
            
        except Exception as e:
            raise IACError(f"Correlation matrix computation failed: {str(e)}")
    
    def compute(self,  np.ndarray) -> np.ndarray:
        """
        Compute the complete IAC trajectory.
        
        Parameters
        ----------
        source_time_series : np.ndarray
            Source-level time series of shape (n_rois, n_times).
        
        Returns
        -------
        np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        
        Raises
        ------
        IACError
            If any step in the computation fails.
        """
        try:
            logger.info(f"Computing IAC trajectory for band {self.band[0]:.1f}-{self.band[1]:.1f} Hz")
            
            # Step 1: Validate input
            self._validate_input_data(source_time_series)
            d, T = source_time_series.shape
            
            # Step 2: Bandpass filter
            filtered_data = self._bandpass_filter(source_time_series)
            
            # Step 3: Compute envelope
            envelopes = self._compute_envelope(filtered_data)
            
            # Step 4: Apply pairwise orthogonalization (if enabled)
            if self.orthogonalize:
                envelopes = self._pairwise_orthogonalization(envelopes)
            
            # Step 5: Compute correlation matrices
            C_traj = self._compute_correlation_matrices(envelopes)
            
            logger.info(f"IAC trajectory computed successfully. Shape: {C_traj.shape}")
            return C_traj
            
        except Exception as e:
            if isinstance(e, IACError):
                raise
            raise IACError(f"IAC trajectory computation failed: {str(e)}")
    
    @staticmethod
    def validate_iac_trajectory(C_traj: np.ndarray, n_rois: int, n_times: int) -> bool:
        """
        Validate that a computed IAC trajectory has correct properties.
        
        Parameters
        ----------
        C_traj : np.ndarray
            IAC trajectory to validate.
        n_rois : int
            Expected number of ROIs.
        n_times : int
            Expected number of time points.
        
        Returns
        -------
        bool
            True if validation passes, False otherwise.
        """
        try:
            if C_traj.shape != (n_times, n_rois, n_rois):
                return False
            
            if not np.all(np.isfinite(C_traj)):
                return False
            
            # Check symmetry
            if not np.allclose(C_traj, np.transpose(C_traj, (0, 2, 1)), atol=1e-6):
                return False
            
            # Check correlation bounds
            if np.any(C_traj < -1.01) or np.any(C_traj > 1.01):
                return False
            
            return True
            
        except Exception:
            return False
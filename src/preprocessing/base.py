import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

# Configure module-specific logger
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing-related errors."""
    pass


class BasePreprocessor(ABC):
    """
    Abstract base class for dataset-specific M/EEG preprocessors.
    
    This class defines the common interface and shared utility methods
    for all dataset preprocessors, ensuring consistent handling of 
    filtering, artifact rejection, and resting-state segment extraction.
    
    Child classes must implement the `load_data` and `extract_resting_segment`
    methods to handle dataset-specific file formats and protocols.
    """
    
    def __init__(
        self,
        data_path: str,
        sfreq: float,
        n_channels: int,
        ch_names: list,
        ch_types: list,
        notch_freq: Optional[float] = 60.0,
        bandpass_freqs: Tuple[float, float] = (1.0, 70.0)
    ):
        """
        Initialize the base preprocessor.
        
        Parameters
        ----------
        data_path : str
            Filesystem path to the raw data file.
        sfreq : float
            Sampling frequency in Hz.
        n_channels : int
            Number of recording channels.
        ch_names : list of str
            Channel names.
        ch_types : list of str
            Channel types (e.g., 'eeg', 'meg').
        notch_freq : float, optional
            Line noise frequency for notch filtering (default: 60.0 Hz).
        bandpass_freqs : tuple of (float, float), optional
            Lower and upper bounds for bandpass filtering (default: (1.0, 70.0) Hz).
        
        Raises
        ------
        PreprocessingError
            If input parameters are invalid.
        """
        if sfreq <= 0:
            raise PreprocessingError("Sampling frequency must be positive.")
        if n_channels <= 0:
            raise PreprocessingError("Number of channels must be positive.")
        if len(ch_names) != n_channels or len(ch_types) != n_channels:
            raise PreprocessingError("Channel names/types length must match n_channels.")
        if bandpass_freqs[0] >= bandpass_freqs[1]:
            raise PreprocessingError("Invalid bandpass frequencies: low >= high.")
        if bandpass_freqs[1] >= sfreq / 2:
            logger.warning(
                f"Upper bandpass frequency {bandpass_freqs[1]} Hz >= Nyquist ({sfreq/2} Hz). "
                "Adjusting to Nyquist."
            )
            bandpass_freqs = (bandpass_freqs[0], sfreq / 2 - 0.1)
        
        self.data_path = data_path
        self.sfreq = float(sfreq)
        self.n_channels = int(n_channels)
        self.ch_names = list(ch_names)
        self.ch_types = list(ch_types)
        self.notch_freq = float(notch_freq) if notch_freq is not None else None
        self.bandpass_freqs = tuple(bandpass_freqs)
        self._raw_data = None  # Will be set by child classes
        
        logger.info(f"Initialized preprocessor for {data_path}")

    @abstractmethod
    def load_data(self) -> np.ndarray:
        """
        Load raw data from file into a NumPy array.
        
        Returns
        -------
        np.ndarray
            Raw data array of shape (n_channels, n_times).
        
        Raises
        ------
        PreprocessingError
            If data cannot be loaded or has invalid dimensions.
        """
        pass

    @abstractmethod
    def extract_resting_segment(self, data: np.ndarray) -> np.ndarray:
        """
        Extract the first 60 seconds of artifact-free, eyes-closed resting state.
        
        Parameters
        ----------
        data : np.ndarray
            Full-length preprocessed data of shape (n_channels, n_times).
        
        Returns
        -------
        np.ndarray
            Resting segment of shape (n_channels, 60 * sfreq).
        
        Raises
        ------
        PreprocessingError
            If resting segment cannot be extracted.
        """
        pass

    def _validate_data_array(self, data: np.ndarray) -> None:
        """
        Validate that the data array has correct shape and finite values.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to validate.
        
        Raises
        ------
        PreprocessingError
            If validation fails.
        """
        if not isinstance(data, np.ndarray):
            raise PreprocessingError("Data must be a NumPy array.")
        if data.ndim != 2:
            raise PreprocessingError(f"Data must be 2D, got {data.ndim}D.")
        if data.shape[0] != self.n_channels:
            raise PreprocessingError(
                f"Channel dimension mismatch: expected {self.n_channels}, got {data.shape[0]}."
            )
        if not np.all(np.isfinite(data)):
            raise PreprocessingError("Data contains non-finite values (NaN/Inf).")
        if data.size == 0:
            raise PreprocessingError("Data array is empty.")

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase bandpass filtering using a 4th-order Butterworth filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_channels, n_times).
        
        Returns
        -------
        np.ndarray
            Filtered data of same shape.
        
        Raises
        ------
        PreprocessingError
            If filtering fails.
        """
        try:
            from scipy.signal import butter, filtfilt
            self._validate_data_array(data)
            
            low, high = self.bandpass_freqs
            nyquist = self.sfreq / 2.0
            if high >= nyquist:
                high = nyquist - 1e-3
            
            # Design Butterworth filter
            b, a = butter(N=4, Wn=[low/nyquist, high/nyquist], btype='band')
            # Apply zero-phase filtering
            filtered = filtfilt(b, a, data, axis=1)
            logger.debug("Bandpass filtering completed.")
            return filtered.astype(np.float64)
            
        except Exception as e:
            raise PreprocessingError(f"Bandpass filtering failed: {str(e)}")

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove line noise.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_channels, n_times).
        
        Returns
        -------
        np.ndarray
            Notch-filtered data of same shape.
        
        Raises
        ------
        PreprocessingError
            If notch filtering fails or notch_freq is None.
        """
        if self.notch_freq is None:
            logger.debug("Notch frequency is None; skipping notch filter.")
            return data
            
        try:
            self._validate_data_array(data)
            from scipy.signal import iirnotch, filtfilt
            
            nyquist = self.sfreq / 2.0
            if self.notch_freq >= nyquist:
                raise PreprocessingError(
                    f"Notch frequency {self.notch_freq} >= Nyquist {nyquist}."
                )
            
            # Design notch filter (Q=30 is standard)
            b, a = iirnotch(w0=self.notch_freq, Q=30.0, fs=self.sfreq)
            notched = filtfilt(b, a, data, axis=1)
            logger.debug("Notch filtering completed.")
            return notched.astype(np.float64)
            
        except Exception as e:
            raise PreprocessingError(f"Notch filtering failed: {str(e)}")

    def run(self) -> np.ndarray:
        """
        Execute the full preprocessing pipeline.
        
        The pipeline consists of:
        1. Loading raw data
        2. Bandpass filtering (1-70 Hz)
        3. Notch filtering (e.g., 60 Hz)
        4. Extracting 60-second resting segment
        
        Returns
        -------
        np.ndarray
            Preprocessed resting-state data of shape (n_channels, 60 * sfreq).
        
        Raises
        ------
        PreprocessingError
            If any step in the pipeline fails.
        """
        try:
            logger.info("Starting preprocessing pipeline...")
            
            # Step 1: Load raw data (implemented by child class)
            raw_data = self.load_data()
            self._validate_data_array(raw_data)
            
            # Step 2: Bandpass filter
            filtered_data = self.bandpass_filter(raw_data)
            
            # Step 3: Notch filter
            notched_data = self.notch_filter(filtered_data)
            
            # Step 4: Extract resting segment (implemented by child class)
            resting_segment = self.extract_resting_segment(notched_data)
            
            # Final validation
            expected_samples = int(60 * self.sfreq)
            if resting_segment.shape[1] != expected_samples:
                raise PreprocessingError(
                    f"Resting segment length {resting_segment.shape[1]} != expected {expected_samples}."
                )
            
            logger.info("Preprocessing completed successfully.")
            return resting_segment.astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Preprocessing pipeline failed: {str(e)}")
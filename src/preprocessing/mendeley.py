# src/preprocessing/mendeley.py

import os
import numpy as np
import logging
from typing import Optional, Tuple, Any
from scipy.io import loadmat
from scipy.signal import find_peaks
from .base import BasePreprocessor, PreprocessingError

# Configure module-specific logger
logger = logging.getLogger(__name__)


class MendeleyOlfactoryPreprocessor(BasePreprocessor):
    """
    Preprocessor for Mendeley Olfactory EEG dataset.
    
    This class handles 4-channel clinical EEG data (Fp1, Fp2, C3, C4)
    recorded at 256 Hz. It implements dataset-specific data loading,
    manual artifact rejection (due to insufficient channels for ICA),
    and extraction of the 60-second pre-odor resting baseline period.
    
    The class supports common file formats (.mat, .csv) and handles
    the low spatial resolution constraints of this dataset.
    """
    
    # Fixed 4-channel montage for Mendeley Olfactory dataset
    MONTAGE_4_CHANNEL = ['Fp1', 'Fp2', 'C3', 'C4']
    
    def __init__(
        self,
        data_path: str,
        sfreq: float = 256.0,
        notch_freq: Optional[float] = 50.0,
        bandpass_freqs: Tuple[float, float] = (1.0, 45.0),
        odor_onset_marker: Optional[str] = 'odor_onset',
        resting_duration: float = 60.0
    ):
        """
        Initialize the Mendeley Olfactory preprocessor.
        
        Parameters
        ----------
        data_path : str
            Path to the EEG data file (.mat or .csv).
        sfreq : float, optional
            Sampling frequency in Hz (default: 256.0).
        notch_freq : float, optional
            Line noise frequency (default: 50.0 for European sites).
        bandpass_freqs : tuple, optional
            Bandpass filter range (default: (1.0, 45.0) Hz, limited by hardware).
        odor_onset_marker : str, optional
            Marker name or value indicating odor onset (default: 'odor_onset').
        resting_duration : float, optional
            Duration of resting baseline to extract in seconds (default: 60.0).
        """
        if not os.path.exists(data_path):
            raise PreprocessingError(f"Data file does not exist: {data_path}")
        
        self.data_path = data_path
        self.odor_onset_marker = odor_onset_marker
        self.resting_duration = float(resting_duration)
        self._file_extension = os.path.splitext(data_path)[1].lower()
        
        # Validate file format
        if self._file_extension not in ['.mat', '.csv']:
            raise PreprocessingError(
                f"Unsupported file format: {self._file_extension}. "
                "Supported formats: .mat, .csv"
            )
        
        # Hardware-limited bandpass (1-45 Hz as specified in methodology)
        if bandpass_freqs[1] > 45.0:
            logger.warning("Mendeley hardware limits bandpass to 45 Hz. Adjusting.")
            bandpass_freqs = (bandpass_freqs[0], 45.0)
        
        super().__init__(
            data_path=data_path,
            sfreq=sfreq,
            n_channels=4,
            ch_names=self.MONTAGE_4_CHANNEL,
            ch_types=['eeg'] * 4,
            notch_freq=notch_freq,
            bandpass_freqs=bandpass_freqs
        )

    def _load_from_mat(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from MATLAB .mat file.
        
        Returns
        -------
        tuple
            (data, events) where events is array of event markers or None.
        """
        try:
            mat_data = loadmat(self.data_path)
            
            # Find data array (typically 4 channels x time)
            data = None
            events = None
            
            # Look for data variables
            for key in ['data', 'eeg', 'signal', 'EEG']:
                if key in mat_
                    data_candidate = mat_data[key]
                    if isinstance(data_candidate, np.ndarray) and data_candidate.ndim == 2:
                        data = data_candidate
                        break
            
            # Look for event markers
            for key in ['events', 'markers', 'event', 'trigger', 'timing']:
                if key in mat_:
                    events = mat_data[key]
                    break
            
            if data is None:
                raise PreprocessingError("No 2D data array found in .mat file.")
            
            # Ensure correct shape (4 channels, time)
            if data.shape[0] == 4:
                pass  # Correct shape
            elif data.shape[1] == 4:
                data = data.T
            else:
                raise PreprocessingError(
                    f"Expected 4 channels, got shape {data.shape}."
                )
            
            return data.astype(np.float64), events
            
        except Exception as e:
            raise PreprocessingError(f"Failed to load .mat file: {str(e)}")

    def _load_from_csv(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load data from CSV file.
        
        Returns
        -------
        tuple
            (data, events) where events is array of event markers or None.
        """
        try:
            import pandas as pd
            
            # Try to read as DataFrame
            df = pd.read_csv(self.data_path)
            
            # Check if first column contains event markers
            if 'event' in df.columns or 'marker' in df.columns or 'trigger' in df.columns:
                # Separate events and data
                event_col = None
                for col in ['event', 'marker', 'trigger']:
                    if col in df.columns:
                        event_col = col
                        break
                
                if event_col:
                    events = df[event_col].values
                    data_cols = [col for col in df.columns if col != event_col]
                    data = df[data_cols].values.T
                else:
                    events = None
                    data = df.values.T
            else:
                events = None
                data = df.values.T
            
            # Validate data shape
            if data.shape[0] not in [4, 5]:  # 4 EEG + 1 event channel possible
                raise PreprocessingError(
                    f"Expected 4 EEG channels, got {data.shape[0]} columns."
                )
            
            if data.shape[0] == 5:
                # Assume last column is events
                events = data[4, :]
                data = data[:4, :]
            
            return data.astype(np.float64), events
            
        except ImportError as e:
            raise PreprocessingError(
                "pandas package required for .csv files. Install with: pip install pandas"
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to load .csv file: {str(e)}")

    def load_data(self) -> np.ndarray:
        """
        Load raw EEG data from file.
        
        Returns
        -------
        np.ndarray
            Raw data array of shape (4, n_times).
        
        Raises
        ------
        PreprocessingError
            If data cannot be loaded or has invalid dimensions.
        """
        logger.info(f"Loading Mendeley Olfactory data from {self.data_path}")
        
        try:
            if self._file_extension == '.mat':
                data, self._events = self._load_from_mat()
            elif self._file_extension == '.csv':
                data, self._events = self._load_from_csv()
            else:
                raise PreprocessingError(f"Unexpected file extension: {self._file_extension}")
            
            # Validate loaded data
            if data.shape[0] != 4:
                raise PreprocessingError(
                    f"Expected 4 channels, got {data.shape[0]} channels."
                )
            
            required_samples = int(self.resting_duration * self.sfreq)
            if data.shape[1] < required_samples:
                raise PreprocessingError(
                    f"Data too short: {data.shape[1]} samples < {required_samples} required."
                )
            
            logger.info(f"Loaded data with shape {data.shape}")
            return data.astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Data loading failed: {str(e)}")

    def _find_odor_onset(self,  np.ndarray) -> int:
        """
        Find odor onset time to extract pre-odor baseline.
        
        Parameters
        ----------
        data : np.ndarray
            EEG data of shape (4, n_times).
        
        Returns
        -------
        int
            Sample index of odor onset, or end of recording if not found.
        """
        required_samples = int(self.resting_duration * self.sfreq)
        
        # Method 1: Use event markers if available
        if hasattr(self, '_events') and self._events is not None:
            events = np.array(self._events).flatten()
            
            # Look for odor onset marker
            if isinstance(self.odor_onset_marker, str):
                # String marker (e.g., 'odor_onset')
                marker_indices = np.where(events == self.odor_onset_marker)[0]
            else:
                # Numerical marker
                marker_indices = np.where(events == self.odor_onset_marker)[0]
            
            if len(marker_indices) > 0:
                odor_onset_sample = marker_indices[0]
                if odor_onset_sample >= required_samples:
                    return odor_onset_sample
                else:
                    logger.warning(
                        f"Odor onset at {odor_onset_sample} samples < {required_samples} required. "
                        "Using end of recording."
                    )
        
        # Method 2: Detect odor onset from EEG artifacts
        # Odor presentation often causes a brief artifact or change in variance
        try:
            # Compute sliding window variance
            window_size = int(0.5 * self.sfreq)  # 500 ms windows
            variance = np.array([
                np.var(data[:, i:i+window_size]) 
                for i in range(0, data.shape[1] - window_size, int(0.1 * self.sfreq))
            ])
            
            # Find significant increase in variance (indicating stimulus artifact)
            peaks, _ = find_peaks(variance, height=np.percentile(variance, 90))
            if len(peaks) > 0:
                odor_onset_time = peaks[0] * 0.1  # Convert to seconds
                odor_onset_sample = int(odor_onset_time * self.sfreq)
                if odor_onset_sample >= required_samples:
                    return odor_onset_sample
        except Exception as e:
            logger.warning(f"Automatic odor onset detection failed: {str(e)}")
        
        # Method 3: Fallback to end of recording
        logger.info("Odor onset not found. Using end of recording for baseline extraction.")
        return data.shape[1]

    def _manual_artifact_rejection(self,  np.ndarray) -> np.ndarray:
        """
        Perform manual artifact rejection suitable for 4-channel EEG.
        
        Since ICA is not feasible with only 4 channels, we use:
        1. Amplitude thresholding
        2. Variance thresholding
        3. Flat channel detection
        
        Parameters
        ----------
        data : np.ndarray
            Filtered EEG data of shape (4, n_times).
        
        Returns
        -------
        np.ndarray
            Data with artifact-contaminated segments zeroed or interpolated.
        """
        try:
            data_clean = data.copy()
            n_channels, n_times = data.shape
            
            # Amplitude thresholds (clinical EEG in microvolts)
            max_amplitude = 200.0  # μV (higher than OpenNeuro due to different equipment)
            min_amplitude = 0.1    # μV (to detect flat channels)
            
            # Variance thresholds
            min_std = 2.0          # μV
            
            # Process in 1-second windows
            window_size = int(self.sfreq)
            n_windows = n_times // window_size
            
            for win in range(n_windows):
                start_idx = win * window_size
                end_idx = start_idx + window_size
                window_data = data[:, start_idx:end_idx]
                
                # Check for excessive amplitude
                if np.any(np.abs(window_data) > max_amplitude):
                    # Zero this window (will be handled in segment selection)
                    data_clean[:, start_idx:end_idx] = 0
                    continue
                
                # Check for flat channels
                channel_stds = np.std(window_data, axis=1)
                if np.any(channel_stds < min_std):
                    data_clean[:, start_idx:end_idx] = 0
                    continue
                
                # Check for excessive line noise at 50 Hz
                if self._has_excessive_50hz_noise(window_data):
                    data_clean[:, start_idx:end_idx] = 0
                    continue
            
            return data_clean.astype(np.float64)
            
        except Exception as e:
            logger.warning(f"Manual artifact rejection failed: {str(e)}. Skipping.")
            return data

    def _has_excessive_50hz_noise(self,  np.ndarray) -> bool:
        """
        Check for excessive 50 Hz line noise in 4-channel EEG.
        
        Parameters
        ----------
        window_data : np.ndarray
            EEG window of shape (4, window_size).
        
        Returns
        -------
        bool
            True if excessive 50 Hz noise is detected.
        """
        if self.notch_freq != 50.0:
            return False
            
        try:
            from scipy.signal import welch
            from scipy.integrate import simps
            
            fs = self.sfreq
            nperseg = min(256, window_data.shape[1])
            
            # Compute PSD
            freqs, psd = welch(window_data, fs=fs, nperseg=nperseg, axis=1)
            
            # Integrate power around 50 Hz
            band_width = 2.0
            low_band = (50.0 - band_width/2, 50.0 + band_width/2)
            band_power = self._integrate_band_power(freqs, psd, low_band)
            
            # Total power in 1-45 Hz
            total_power = self._integrate_band_power(freqs, psd, (1.0, 45.0))
            
            if total_power == 0:
                return False
                
            noise_ratio = band_power / total_power
            return noise_ratio > 0.4  # 40% threshold for low-channel EEG
            
        except Exception as e:
            logger.warning(f"50 Hz noise detection failed: {str(e)}. Skipping check.")
            return False

    def _integrate_band_power(self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
        """Integrate PSD over a frequency band."""
        low, high = band
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            return 0.0
        return simps(psd[:, idx].mean(axis=0), freqs[idx])

    def extract_resting_segment(self,  np.ndarray) -> np.ndarray:
        """
        Extract the 60-second pre-odor resting baseline segment.
        
        For Mendeley Olfactory dataset, we:
        1. Find odor onset time using event markers or automatic detection
        2. Extract the 60 seconds immediately preceding odor onset
        3. Apply manual artifact rejection (no ICA possible with 4 channels)
        4. Return the cleanest available segment
        
        Parameters
        ----------
        data : np.ndarray
            Filtered EEG data of shape (4, n_times).
        
        Returns
        -------
        np.ndarray
            Resting segment of shape (4, 60 * sfreq).
        
        Raises
        ------
        PreprocessingError
            If no valid segment can be extracted.
        """
        try:
            self._validate_data_array(data)
            
            required_samples = int(self.resting_duration * self.sfreq)
            
            # Step 1: Find odor onset
            odor_onset_sample = self._find_odor_onset(data)
            
            # Step 2: Extract pre-odor baseline
            baseline_end = odor_onset_sample
            baseline_start = baseline_end - required_samples
            
            if baseline_start < 0:
                raise PreprocessingError(
                    f"Insufficient pre-odor data: need {required_samples} samples, "
                    f"available: {odor_onset_sample}."
                )
            
            resting_segment = data[:, baseline_start:baseline_end].copy()
            
            # Step 3: Apply manual artifact rejection
            clean_segment = self._manual_artifact_rejection(resting_segment)
            
            # Step 4: Validate final segment
            if clean_segment.shape[1] != required_samples:
                raise PreprocessingError(
                    f"Segment length mismatch: expected {required_samples}, got {clean_segment.shape[1]}."
                )
            
            # Check if segment is mostly zero (completely artifact-contaminated)
            zero_fraction = np.mean(clean_segment == 0)
            if zero_fraction > 0.5:
                logger.warning(
                    f"Segment is {zero_fraction*100:.1f}% zero-padded due to artifacts. "
                    "Consider manual inspection."
                )
            
            logger.info(f"Extracted pre-odor baseline from {baseline_start} to {baseline_end} samples.")
            return clean_segment.astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Resting segment extraction failed: {str(e)}")
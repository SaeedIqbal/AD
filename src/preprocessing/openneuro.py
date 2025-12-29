import os
import numpy as np
import logging
from typing import Optional, Tuple, Any
from scipy.io import loadmat
from .base import BasePreprocessor, PreprocessingError

# Configure module-specific logger
logger = logging.getLogger(__name__)


class OpenNeuroPreprocessor(BasePreprocessor):
    """
    Preprocessor for OpenNeuro ds004504 dataset.
    
    This class handles 19-channel clinical EEG data recorded with the 
    10-20 system at 500 Hz. It implements dataset-specific data loading,
    ICA-based artifact removal, and extraction of the first 60 seconds 
    of eyes-closed resting state.
    
    The class supports common EEG file formats (.set, .edf, .mat) found 
    in the OpenNeuro repository.
    """
    
    # Standard 10-20 montage for 19 channels
    STANDARD_1020 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
    ]
    
    def __init__(
        self,
        data_path: str,
        sfreq: float = 500.0,
        notch_freq: Optional[float] = 60.0,
        bandpass_freqs: Tuple[float, float] = (1.0, 70.0),
        ica_n_components: Optional[float] = 0.99,
        random_state: int = 42
    ):
        """
        Initialize the OpenNeuro preprocessor.
        
        Parameters
        ----------
        data_path : str
            Path to the EEG data file (.set, .edf, or .mat).
        sfreq : float, optional
            Sampling frequency in Hz (default: 500.0).
        notch_freq : float, optional
            Line noise frequency (default: 60.0 for North American sites).
        bandpass_freqs : tuple, optional
            Bandpass filter range (default: (1.0, 70.0) Hz).
        ica_n_components : float, optional
            Number of ICA components to retain (default: 0.99 for 99% variance).
        random_state : int, optional
            Random seed for ICA reproducibility (default: 42).
        """
        if not os.path.exists(data_path):
            raise PreprocessingError(f"Data file does not exist: {data_path}")
        
        self.data_path = data_path
        self.ica_n_components = ica_n_components
        self.random_state = random_state
        self._file_extension = os.path.splitext(data_path)[1].lower()
        
        # Validate file format
        if self._file_extension not in ['.set', '.edf', '.mat']:
            raise PreprocessingError(
                f"Unsupported file format: {self._file_extension}. "
                "Supported formats: .set, .edf, .mat"
            )
        
        super().__init__(
            data_path=data_path,
            sfreq=sfreq,
            n_channels=19,
            ch_names=self.STANDARD_1020,
            ch_types=['eeg'] * 19,
            notch_freq=notch_freq,
            bandpass_freqs=bandpass_freqs
        )

    def _load_from_set(self) -> np.ndarray:
        """Load data from EEGLAB .set file."""
        try:
            import mne
            raw = mne.io.read_raw_eeglab(self.data_path, preload=True)
            return raw.get_data()
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for .set files. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to load .set file: {str(e)}")

    def _load_from_edf(self) -> np.ndarray:
        """Load data from EDF file."""
        try:
            import mne
            raw = mne.io.read_raw_edf(self.data_path, preload=True)
            return raw.get_data()
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for .edf files. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to load .edf file: {str(e)}")

    def _load_from_mat(self) -> np.ndarray:
        """Load data from MATLAB .mat file."""
        try:
            mat_data = loadmat(self.data_path)
            
            # Try common variable names
            data = None
            for key in ['data', 'eeg_data', 'EEG', 'signal']:
                if key in mat_data:
                    data = mat_data[key]
                    break
            
            if data is None:
                # Use first array-like variable
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        data = value
                        break
            
            if data is None:
                raise PreprocessingError("No 2D data array found in .mat file.")
            
            # Ensure correct shape (channels, time)
            if data.shape[0] == 19:
                return data.astype(np.float64)
            elif data.shape[1] == 19:
                return data.T.astype(np.float64)
            else:
                raise PreprocessingError(
                    f"Expected 19 channels, got shape {data.shape}."
                )
                
        except Exception as e:
            raise PreprocessingError(f"Failed to load .mat file: {str(e)}")

    def load_data(self) -> np.ndarray:
        """
        Load raw EEG data from file.
        
        Returns
        -------
        np.ndarray
            Raw data array of shape (19, n_times).
        
        Raises
        ------
        PreprocessingError
            If data cannot be loaded or has invalid dimensions.
        """
        logger.info(f"Loading OpenNeuro data from {self.data_path}")
        
        try:
            if self._file_extension == '.set':
                data = self._load_from_set()
            elif self._file_extension == '.edf':
                data = self._load_from_edf()
            elif self._file_extension == '.mat':
                data = self._load_from_mat()
            else:
                raise PreprocessingError(f"Unexpected file extension: {self._file_extension}")
            
            # Validate loaded data
            if data.shape[0] != 19:
                # Try transposing
                if data.shape[1] == 19:
                    data = data.T
                else:
                    raise PreprocessingError(
                        f"Expected 19 channels, got {data.shape[0]} channels."
                    )
            
            if data.shape[1] < 60 * self.sfreq:
                raise PreprocessingError(
                    f"Data too short: {data.shape[1]} samples < {60 * self.sfreq} required."
                )
            
            logger.info(f"Loaded data with shape {data.shape}")
            return data.astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Data loading failed: {str(e)}")

    def _apply_ica(self,  np.ndarray) -> np.ndarray:
        """
        Apply ICA for artifact removal (eye blink, muscle, cardiac).
        
        Parameters
        ----------
        data : np.ndarray
            Filtered EEG data of shape (19, n_times).
        
        Returns
        -------
        np.ndarray
            ICA-corrected data of same shape.
        """
        try:
            import mne
            from mne.preprocessing import ICA
            
            # Create MNE Raw object for ICA
            info = mne.create_info(
                ch_names=self.ch_names,
                sfreq=self.sfreq,
                ch_types='eeg'
            )
            raw = mne.io.RawArray(data, info, copy='auto')
            
            # Set EEG reference
            raw.set_eeg_reference('average', projection=True)
            
            # Fit ICA
            ica = ICA(
                n_components=self.ica_n_components,
                random_state=self.random_state,
                max_iter='auto'
            )
            ica.fit(raw)
            
            # Automatically detect EOG components
            try:
                eog_indices, _ = ica.find_bads_eog(raw)
                ica.exclude = eog_indices
            except Exception as e:
                logger.warning(f"EOG detection failed: {str(e)}. Skipping automatic exclusion.")
                ica.exclude = []
            
            # Apply ICA
            corrected_raw = ica.apply(raw)
            corrected_data = corrected_raw.get_data()
            
            logger.info(f"ICA applied. Excluded components: {ica.exclude}")
            return corrected_data.astype(np.float64)
            
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for ICA. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"ICA processing failed: {str(e)}")

    def extract_resting_segment(self,  np.ndarray) -> np.ndarray:
        """
        Extract the first 60 seconds of eyes-closed resting state.
        
        For OpenNeuro ds004504, we assume the recording starts with 
        eyes-closed resting state. Manual artifact rejection is performed 
        by selecting the first continuous 60-second segment that passes 
        basic amplitude thresholds.
        
        Parameters
        ----------
        data : np.ndarray
            Preprocessed EEG data of shape (19, n_times).
        
        Returns
        -------
        np.ndarray
            Resting segment of shape (19, 60 * sfreq).
        
        Raises
        ------
        PreprocessingError
            If no valid 60-second segment can be found.
        """
        try:
            self._validate_data_array(data)
            
            # Apply ICA for artifact removal
            ica_corrected = self._apply_ica(data)
            
            # Define amplitude thresholds (microvolts)
            # Clinical EEG typically has amplitudes < 100 μV
            max_amplitude = 150.0  # μV
            min_std = 1.0          # μV (to exclude flat channels)
            
            n_samples = int(60 * self.sfreq)
            n_total = ica_corrected.shape[1]
            
            if n_total < n_samples:
                raise PreprocessingError(
                    f"Insufficient data: {n_total} samples < {n_samples} required."
                )
            
            # Search for valid 60-second segment starting from beginning
            max_start = n_total - n_samples
            for start_idx in range(0, max_start + 1, int(self.sfreq)):  # Check every second
                end_idx = start_idx + n_samples
                segment = ica_corrected[:, start_idx:end_idx]
                
                # Check amplitude thresholds
                if np.any(np.abs(segment) > max_amplitude):
                    continue
                
                # Check for flat channels (low standard deviation)
                channel_stds = np.std(segment, axis=1)
                if np.any(channel_stds < min_std):
                    continue
                
                # Check for excessive line noise (60 Hz + harmonics)
                if self._has_excessive_line_noise(segment):
                    continue
                
                logger.info(f"Found valid resting segment at {start_idx / self.sfreq:.1f} seconds.")
                return segment.astype(np.float64)
            
            # If no segment found, return first 60 seconds with warning
            logger.warning("No artifact-free segment found. Returning first 60 seconds.")
            return ica_corrected[:, :n_samples].astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Resting segment extraction failed: {str(e)}")

    def _has_excessive_line_noise(self,  np.ndarray) -> bool:
        """
        Check if segment has excessive line noise at 60 Hz and harmonics.
        
        Parameters
        ----------
        segment : np.ndarray
            EEG segment of shape (19, n_samples).
        
        Returns
        -------
        bool
            True if excessive line noise is detected.
        """
        if self.notch_freq is None:
            return False
            
        try:
            from scipy.signal import welch
            from scipy.integrate import simps
            
            fs = self.sfreq
            nperseg = min(1024, segment.shape[1])
            
            # Compute power spectral density
            freqs, psd = welch(segment, fs=fs, nperseg=nperseg, axis=1)
            
            # Check power at line frequency and first harmonic
            line_idx = np.argmin(np.abs(freqs - self.notch_freq))
            harmonic_idx = np.argmin(np.abs(freqs - 2 * self.notch_freq))
            
            # Integrate power in narrow bands around line frequency
            band_width = 2.0  # Hz
            line_band = (self.notch_freq - band_width/2, self.notch_freq + band_width/2)
            harmonic_band = (2 * self.notch_freq - band_width/2, 2 * self.notch_freq + band_width/2)
            
            line_power = self._integrate_band_power(freqs, psd, line_band)
            harmonic_power = self._integrate_band_power(freqs, psd, harmonic_band)
            
            # Total power in 1-70 Hz range
            total_power = self._integrate_band_power(freqs, psd, (1.0, 70.0))
            
            # Check if line noise power exceeds threshold
            if total_power == 0:
                return False
                
            line_ratio = (line_power + harmonic_power) / total_power
            return line_ratio > 0.3  # 30% threshold
            
        except Exception as e:
            logger.warning(f"Line noise detection failed: {str(e)}. Skipping check.")
            return False

    def _integrate_band_power(self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
        """Integrate PSD over a frequency band."""
        low, high = band
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            return 0.0
        return simps(psd[:, idx].mean(axis=0), freqs[idx])
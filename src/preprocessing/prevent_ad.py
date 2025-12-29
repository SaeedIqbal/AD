import os
import numpy as np
import logging
from typing import Optional, Tuple, Any
from .base import BasePreprocessor, PreprocessingError

# Configure module-specific logger
logger = logging.getLogger(__name__)


class PREVENTADPreprocessor(BasePreprocessor):
    """
    Preprocessor for PREVENT-AD EEG dataset.
    
    This class handles two subgroups of EEG data:
    - 64-channel BrainAmp system at 500 Hz
    - 32-channel g.tec system at 256 Hz
    Both use extended 10-20 montages. The class implements dataset-specific 
    data loading, ICA-based artifact removal, and extraction of a unified 
    60-second eyes-closed resting segment.
    
    The class supports common EEG file formats (.vhdr/.eeg, .gdf, .edf) 
    used in the PREVENT-AD repository.
    """
    
    # Extended 10-20 montages for PREVENT-AD subgroups
    BRAINAMP_64_CHANNELS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3',
        'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
        'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4',
        'PO8', 'FT9', 'FT7', 'FT8', 'FT10', 'TP7', 'TP8'
    ]
    
    GTEC_32_CHANNELS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ]
    
    def __init__(
        self,
        data_path: str,
        system_type: str = 'auto',
        notch_freq: Optional[float] = 60.0,
        bandpass_freqs: Tuple[float, float] = (1.0, 70.0),
        ica_n_components: Optional[float] = 0.99,
        random_state: int = 42
    ):
        """
        Initialize the PREVENT-AD preprocessor.
        
        Parameters
        ----------
        data_path : str
            Path to the EEG data file (.vhdr, .gdf, .edf, etc.).
        system_type : str, optional
            EEG system type: 'brainamp', 'gtec', or 'auto' (default: 'auto').
        notch_freq : float, optional
            Line noise frequency (default: 60.0).
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
        self.system_type = system_type.lower()
        self.ica_n_components = ica_n_components
        self.random_state = random_state
        self._file_extension = os.path.splitext(data_path)[1].lower()
        
        # Validate file format
        supported_formats = ['.vhdr', '.eeg', '.gdf', '.edf', '.bdf']
        if self._file_extension not in supported_formats:
            raise PreprocessingError(
                f"Unsupported file format: {self._file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        # Auto-detect system type if not specified
        if self.system_type == 'auto':
            self.system_type = self._detect_system_type()
        
        if self.system_type not in ['brainamp', 'gtec']:
            raise PreprocessingError(
                f"Invalid system type: {self.system_type}. Must be 'brainamp', 'gtec', or 'auto'."
            )
        
        # Set system-specific parameters
        if self.system_type == 'brainamp':
            sfreq = 500.0
            n_channels = 64
            ch_names = self.BRAINAMP_64_CHANNELS
        else:  # gtec
            sfreq = 256.0
            n_channels = 32
            ch_names = self.GTEC_32_CHANNELS
        
        super().__init__(
            data_path=data_path,
            sfreq=sfreq,
            n_channels=n_channels,
            ch_names=ch_names,
            ch_types=['eeg'] * n_channels,
            notch_freq=notch_freq,
            bandpass_freqs=bandpass_freqs
        )

    def _detect_system_type(self) -> str:
        """
        Auto-detect EEG system type based on file characteristics.
        
        Returns
        -------
        str
            Detected system type ('brainamp' or 'gtec').
        """
        try:
            import mne
            
            # Try to read file header to get sampling frequency
            if self._file_extension == '.vhdr':
                # Brainstorm format typically uses 500 Hz
                raw = mne.io.read_raw_brainstorm(self.data_path, preload=False)
            elif self._file_extension == '.gdf':
                # g.tec systems often use 256 Hz
                raw = mne.io.read_raw_gdf(self.data_path, preload=False)
            elif self._file_extension in ['.edf', '.bdf']:
                raw = mne.io.read_raw_edf(self.data_path, preload=False)
            else:
                # Default to brainamp for .eeg files
                return 'brainamp'
            
            sfreq = raw.info['sfreq']
            n_channels = len(raw.ch_names)
            
            # Heuristic: 500 Hz + 64 channels = BrainAmp, 256 Hz + 32 channels = g.tec
            if abs(sfreq - 500.0) < 10 and n_channels >= 60:
                return 'brainamp'
            elif abs(sfreq - 256.0) < 10 and n_channels >= 30:
                return 'gtec'
            else:
                logger.warning(
                    f"Unclear system type: sfreq={sfreq}, n_channels={n_channels}. "
                    "Defaulting to brainamp."
                )
                return 'brainamp'
                
        except Exception as e:
            logger.warning(f"System type detection failed: {str(e)}. Defaulting to brainamp.")
            return 'brainamp'

    def _load_brainstorm_data(self) -> np.ndarray:
        """Load data from Brainstorm .vhdr/.eeg files."""
        try:
            import mne
            raw = mne.io.read_raw_brainstorm(self.data_path, preload=True)
            return raw.get_data()
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for Brainstorm files. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to load Brainstorm file: {str(e)}")

    def _load_gdf_data(self) -> np.ndarray:
        """Load data from g.tec .gdf files."""
        try:
            import mne
            raw = mne.io.read_raw_gdf(self.data_path, preload=True)
            return raw.get_data()
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for GDF files. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to load GDF file: {str(e)}")

    def _load_edf_data(self) -> np.ndarray:
        """Load data from EDF/BDF files."""
        try:
            import mne
            if self._file_extension == '.bdf':
                raw = mne.io.read_raw_bdf(self.data_path, preload=True)
            else:
                raw = mne.io.read_raw_edf(self.data_path, preload=True)
            return raw.get_data()
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for EDF/BDF files. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"Failed to load EDF/BDF file: {str(e)}")

    def load_data(self) -> np.ndarray:
        """
        Load raw EEG data from file.
        
        Returns
        -------
        np.ndarray
            Raw data array of shape (n_channels, n_times).
        
        Raises
        ------
        PreprocessingError
            If data cannot be loaded or has invalid dimensions.
        """
        logger.info(f"Loading PREVENT-AD data from {self.data_path} ({self.system_type})")
        
        try:
            # Load data based on file format
            if self._file_extension == '.vhdr':
                data = self._load_brainstorm_data()
            elif self._file_extension == '.gdf':
                data = self._load_gdf_data()
            elif self._file_extension in ['.edf', '.bdf', '.eeg']:
                data = self._load_edf_data()
            else:
                raise PreprocessingError(f"Unexpected file extension: {self._file_extension}")
            
            # Validate loaded data
            expected_channels = 64 if self.system_type == 'brainamp' else 32
            if data.shape[0] != expected_channels:
                # Try to match channels by name if possible
                if hasattr(self, '_raw'):
                    actual_names = self._raw.ch_names
                    target_names = self.BRAINAMP_64_CHANNELS if self.system_type == 'brainamp' else self.GTEC_32_CHANNELS
                    
                    # Find intersection of channels
                    common_channels = [ch for ch in target_names if ch in actual_names]
                    if len(common_channels) >= expected_channels * 0.8:  # 80% overlap acceptable
                        logger.warning(f"Channel mismatch. Using {len(common_channels)} common channels.")
                        data = data[[actual_names.index(ch) for ch in common_channels]]
                    else:
                        raise PreprocessingError(
                            f"Channel mismatch: expected {expected_channels}, got {data.shape[0]}."
                        )
                else:
                    raise PreprocessingError(
                        f"Channel mismatch: expected {expected_channels}, got {data.shape[0]}."
                    )
            
            required_samples = int(60 * self.sfreq)
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

    def _apply_ica(self,  np.ndarray) -> np.ndarray:
        """
        Apply ICA for artifact removal (eye blink, muscle, cardiac).
        
        Parameters
        ----------
        data : np.ndarray
            Filtered EEG data of shape (n_channels, n_times).
        
        Returns
        -------
        np.ndarray
            ICA-corrected data of same shape.
        """
        try:
            import mne
            from mne.preprocessing import ICA
            
            # Create MNE Raw object
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
                max_iter='auto',
                method='fastica'
            )
            ica.fit(raw)
            
            # Automatically detect EOG components
            try:
                # Try common EOG channel names
                eog_ch_names = ['EOG1', 'EOG2', 'VEOG', 'HEOG', 'AUX1', 'AUX2']
                eog_indices = []
                
                for eog_ch in eog_ch_names:
                    if eog_ch in raw.ch_names:
                        eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
                        eog_indices.extend(eog_idx)
                
                if eog_indices:
                    ica.exclude = list(set(eog_indices))  # Remove duplicates
                else:
                    # No EOG channels found, use correlation-based detection
                    raise ValueError("No EOG channels available")
                    
            except Exception:
                # Fallback: exclude components with high correlation to frontal channels
                logger.warning("EOG detection failed. Using frontal correlation method.")
                frontal_channels = [i for i, ch in enumerate(self.ch_names) if 'F' in ch]
                if frontal_channels:
                    sources = ica.get_sources(raw).get_data()
                    correlations = np.array([
                        np.max(np.abs(np.corrcoef(sources[i], data[frontal_channels].mean(axis=0))[0, 1:]))
                        for i in range(sources.shape[0])
                    ])
                    # Exclude top 10% most correlated components
                    threshold = np.percentile(correlations, 90)
                    ica.exclude = np.where(correlations > threshold)[0].tolist()
                else:
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
        Extract a unified 60-second eyes-closed resting segment.
        
        For PREVENT-AD, we:
        1. Apply ICA for comprehensive artifact removal
        2. Search for the cleanest 60-second segment in the first 5 minutes
        3. Use amplitude and variance thresholds for quality control
        4. Ensure segment represents true resting state
        
        Parameters
        ----------
        data : np.ndarray
            Filtered EEG data of shape (n_channels, n_times).
        
        Returns
        -------
        np.ndarray
            Resting segment of shape (n_channels, 60 * sfreq).
        
        Raises
        ------
        PreprocessingError
            If no valid segment can be extracted.
        """
        try:
            self._validate_data_array(data)
            
            required_samples = int(60 * self.sfreq)
            n_total = data.shape[1]
            
            if n_total < required_samples:
                raise PreprocessingError(
                    f"Insufficient  {n_total} samples < {required_samples} required."
                )
            
            # Step 1: Apply ICA
            ica_corrected = self._apply_ica(data)
            
            # Step 2: Define search window (first 5 minutes to ensure resting state)
            search_end = min(n_total, int(300 * self.sfreq))  # 300 seconds = 5 minutes
            if search_end < required_samples:
                search_end = n_total
            
            # Step 3: Find best segment using quality metrics
            max_amplitude = 150.0  # μV
            min_std = 1.5          # μV
            best_segment = None
            best_score = -float('inf')
            
            # Sample every 2 seconds to balance quality and computation
            step_size = int(2 * self.sfreq)
            start_points = range(0, search_end - required_samples + 1, step_size)
            
            for start_idx in start_points:
                end_idx = start_idx + required_samples
                segment = ica_corrected[:, start_idx:end_idx]
                
                # Quality checks
                if np.any(np.abs(segment) > max_amplitude):
                    continue
                
                channel_stds = np.std(segment, axis=1)
                if np.any(channel_stds < min_std):
                    continue
                
                # Quality score: higher variance = better signal quality
                quality_score = np.mean(channel_stds)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_segment = segment
            
            if best_segment is not None:
                logger.info(f"Found best resting segment with quality score: {best_score:.2f}")
                return best_segment.astype(np.float64)
            else:
                # Fallback: return first 60 seconds
                logger.warning("No high-quality segment found. Returning first 60 seconds.")
                return ica_corrected[:, :required_samples].astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Resting segment extraction failed: {str(e)}")
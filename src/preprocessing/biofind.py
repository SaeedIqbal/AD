import os
import numpy as np
import logging
from typing import Optional, Tuple, Any
from .base import BasePreprocessor, PreprocessingError

# Configure module-specific logger
logger = logging.getLogger(__name__)


class BioFINDPreprocessor(BasePreprocessor):
    """
    Preprocessor for BioFIND MEG dataset.
    
    This class handles 306-channel MEG data (102 magnetometers + 204 gradiometers)
    recorded at 1000 Hz. It implements dataset-specific data loading,
    tSSS for external interference suppression, ICA for biological artifacts,
    SOUND algorithm for sensor-level denoising, and extraction of a 
    60-second artifact-free resting segment.
    
    The class requires MNE-Python for MEG-specific processing.
    """
    
    def __init__(
        self,
        data_path: str,
        sfreq: float = 1000.0,
        notch_freq: Optional[float] = 60.0,
        bandpass_freqs: Tuple[float, float] = (1.0, 70.0),
        tsss_duration: float = 10.0,
        tsss_st_duration: float = 0.1,
        ica_n_components: Optional[int] = 25,
        sound_n_neighbors: int = 30,
        random_state: int = 42
    ):
        """
        Initialize the BioFIND preprocessor.
        
        Parameters
        ----------
        data_path : str
            Path to the MEG data file (.fif format).
        sfreq : float, optional
            Sampling frequency in Hz (default: 1000.0).
        notch_freq : float, optional
            Line noise frequency (default: 60.0).
        bandpass_freqs : tuple, optional
            Bandpass filter range (default: (1.0, 70.0) Hz).
        tsss_duration : float, optional
            Duration for tSSS buffer in seconds (default: 10.0).
        tsss_st_duration : float, optional
            Short-term buffer duration for tSSS (default: 0.1 s).
        ica_n_components : int, optional
            Number of ICA components to compute (default: 25).
        sound_n_neighbors : int, optional
            Number of neighbors for SOUND algorithm (default: 30).
        random_state : int, optional
            Random seed for ICA reproducibility (default: 42).
        """
        if not os.path.exists(data_path):
            raise PreprocessingError(f"Data file does not exist: {data_path}")
        
        if not data_path.endswith('.fif'):
            raise PreprocessingError(f"BioFIND requires .fif files, got: {data_path}")
        
        self.data_path = data_path
        self.tsss_duration = float(tsss_duration)
        self.tsss_st_duration = float(tsss_st_duration)
        self.ica_n_components = int(ica_n_components) if ica_n_components else None
        self.sound_n_neighbors = int(sound_n_neighbors)
        self.random_state = int(random_state)
        
        # BioFIND has 306 channels: 102 magnetometers + 204 gradiometers
        n_channels = 306
        ch_names = [f'MAG{i:03d}' for i in range(102)] + [f'GRAD{i:03d}' for i in range(204)]
        ch_types = ['mag'] * 102 + ['grad'] * 204
        
        super().__init__(
            data_path=data_path,
            sfreq=sfreq,
            n_channels=n_channels,
            ch_names=ch_names,
            ch_types=ch_types,
            notch_freq=notch_freq,
            bandpass_freqs=bandpass_freqs
        )

    def load_data(self) -> np.ndarray:
        """
        Load raw MEG data from .fif file.
        
        Returns
        -------
        np.ndarray
            Raw data array of shape (306, n_times).
        
        Raises
        ------
        PreprocessingError
            If data cannot be loaded or has invalid dimensions.
        """
        logger.info(f"Loading BioFIND MEG data from {self.data_path}")
        
        try:
            import mne
            
            # Load raw MEG data
            raw = mne.io.read_raw_fif(self.data_path, preload=True)
            
            # Extract data as NumPy array
            data = raw.get_data()
            
            # Validate channel count
            if data.shape[0] != 306:
                raise PreprocessingError(
                    f"Expected 306 MEG channels, got {data.shape[0]}."
                )
            
            # Validate sampling frequency
            if abs(raw.info['sfreq'] - self.sfreq) > 0.1:
                logger.warning(
                    f"Expected sfreq {self.sfreq}, got {raw.info['sfreq']}. "
                    "Using actual sampling rate."
                )
                self.sfreq = raw.info['sfreq']
            
            if data.shape[1] < 60 * self.sfreq:
                raise PreprocessingError(
                    f"Data too short: {data.shape[1]} samples < {60 * self.sfreq} required."
                )
            
            logger.info(f"Loaded MEG data with shape {data.shape}")
            return data.astype(np.float64)
            
        except ImportError as e:
            raise PreprocessingError(
                "mne package required for MEG processing. Install with: pip install mne"
            )
        except Exception as e:
            raise PreprocessingError(f"MEG data loading failed: {str(e)}")

    def _apply_tsss(self,  np.ndarray) -> np.ndarray:
        """
        Apply Temporal Signal Space Separation (tSSS) for external interference suppression.
        
        Parameters
        ----------
        data : np.ndarray
            Raw MEG data of shape (306, n_times).
        
        Returns
        -------
        np.ndarray
            tSSS-corrected data of same shape.
        """
        try:
            import mne
            
            # Create MNE Raw object
            info = mne.create_info(
                ch_names=self.ch_names,
                sfreq=self.sfreq,
                ch_types=self.ch_types
            )
            raw = mne.io.RawArray(data, info, copy='auto')
            
            # Apply tSSS
            raw_tsss = mne.preprocessing.maxwell_filter(
                raw,
                origin=(0., 0., 0.04),  # Head origin in meters
                int_order=8,            # Internal expansion order
                ext_order=3,            # External expansion order
                calibration=None,       # No calibration file
                cross_talk=None,        # No cross-talk file
                st_duration=self.tsss_st_duration,
                st_correlation=0.98,
                verbose=False
            )
            
            tsss_data = raw_tsss.get_data()
            logger.info("tSSS applied successfully.")
            return tsss_data.astype(np.float64)
            
        except Exception as e:
            raise PreprocessingError(f"tSSS processing failed: {str(e)}")

    def _apply_sound_denoising(self,  np.ndarray) -> np.ndarray:
        """
        Apply SOUND algorithm for sensor-level denoising.
        
        Parameters
        ----------
        data : np.ndarray
            tSSS-corrected MEG data of shape (306, n_times).
        
        Returns
        -------
        np.ndarray
            SOUND-denoised data of same shape.
        """
        try:
            from scipy.spatial.distance import pdist, squareform
            from scipy.linalg import svd
            
            n_channels, n_times = data.shape
            n_neighbors = min(self.sound_n_neighbors, n_channels - 1)
            
            # Compute channel correlation matrix
            corr_matrix = np.corrcoef(data)
            np.fill_diagonal(corr_matrix, 0)  # Remove self-correlation
            
            # Find nearest neighbors for each channel
            denoised_data = np.zeros_like(data)
            
            for ch in range(n_channels):
                # Get correlation scores for this channel
                corr_scores = corr_matrix[ch]
                
                # Find n_neighbors with highest absolute correlation
                neighbor_indices = np.argpartition(np.abs(corr_scores), -n_neighbors)[-n_neighbors:]
                
                # Extract neighbor data
                neighbor_data = data[neighbor_indices]
                
                # Perform SVD on neighbor data
                U, s, Vt = svd(neighbor_data, full_matrices=False)
                
                # Reconstruct using dominant components (keep 95% of energy)
                energy_cumsum = np.cumsum(s**2)
                energy_total = energy_cumsum[-1]
                if energy_total > 0:
                    n_components = np.searchsorted(energy_cumsum, 0.95 * energy_total) + 1
                    n_components = max(1, min(n_components, len(s)))
                    
                    # Reconstruct
                    reconstructed = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
                    denoised_data[ch] = reconstructed[0]  # Use first neighbor's reconstruction
                else:
                    denoised_data[ch] = data[ch]
            
            logger.info("SOUND denoising applied successfully.")
            return denoised_data.astype(np.float64)
            
        except Exception as e:
            logger.warning(f"SOUND denoising failed: {str(e)}. Skipping denoising.")
            return data

    def _apply_ica(self,  np.ndarray) -> np.ndarray:
        """
        Apply ICA for biological artifact removal.
        
        Parameters
        ----------
        data : np.ndarray
            Preprocessed MEG data of shape (306, n_times).
        
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
                ch_types=self.ch_types
            )
            raw = mne.io.RawArray(data, info, copy='auto')
            
            # Fit ICA
            ica = ICA(
                n_components=self.ica_n_components,
                random_state=self.random_state,
                max_iter='auto',
                method='fastica'
            )
            ica.fit(raw)
            
            # For MEG, we typically exclude components with high ECG correlation
            # Since BioFIND includes ECG, we can use it for automatic detection
            try:
                # Check if ECG channel exists (BioFIND typically has it)
                ecg_indices, _ = ica.find_bads_ecg(raw, ch_name='EEG063')  # Common ECG channel name
                ica.exclude = ecg_indices
            except Exception:
                # If ECG not available, exclude components with high variance
                logger.warning("ECG channel not found. Using variance-based component exclusion.")
                component_vars = np.var(ica.get_sources(raw).get_data(), axis=1)
                # Exclude top 5% most variable components
                threshold = np.percentile(component_vars, 95)
                ica.exclude = np.where(component_vars > threshold)[0].tolist()
            
            # Apply ICA
            corrected_raw = ica.apply(raw)
            corrected_data = corrected_raw.get_data()
            
            logger.info(f"ICA applied. Excluded components: {ica.exclude}")
            return corrected_data.astype(np.float64)
            
        except Exception as e:
            raise PreprocessingError(f"ICA processing failed: {str(e)}")

    def extract_resting_segment(self,  np.ndarray) -> np.ndarray:
        """
        Extract a 60-second artifact-free resting segment.
        
        For BioFIND, we implement a multi-stage artifact detection:
        1. Apply tSSS for external interference
        2. Apply SOUND for sensor-level denoising  
        3. Apply ICA for biological artifacts
        4. Search for 60-second segment with minimal residual artifacts
        
        Parameters
        ----------
        data : np.ndarray
            Filtered MEG data of shape (306, n_times).
        
        Returns
        -------
        np.ndarray
            Resting segment of shape (306, 60 * sfreq).
        
        Raises
        ------
        PreprocessingError
            If no valid 60-second segment can be found.
        """
        try:
            self._validate_data_array(data)
            
            n_samples = int(60 * self.sfreq)
            n_total = data.shape[1]
            
            if n_total < n_samples:
                raise PreprocessingError(
                    f"Insufficient data: {n_total} samples < {n_samples} required."
                )
            
            # Step 1: Apply tSSS
            tsss_data = self._apply_tsss(data)
            
            # Step 2: Apply SOUND denoising
            sound_data = self._apply_sound_denoising(tsss_data)
            
            # Step 3: Apply ICA
            ica_data = self._apply_ica(sound_data)
            
            # Step 4: Find artifact-free 60-second segment
            # Use variance thresholding to detect artifacts
            max_variance = 1e-20  # T^2, typical for clean MEG
            min_variance = 1e-28  # T^2, to exclude dead channels
            
            max_start = n_total - n_samples
            best_segment = None
            min_artifact_score = float('inf')
            
            # Sample potential start points every 5 seconds to reduce computation
            start_points = range(0, max_start + 1, int(5 * self.sfreq))
            if not start_points:
                start_points = [0]
            
            for start_idx in start_points:
                end_idx = start_idx + n_samples
                segment = ica_data[:, start_idx:end_idx]
                
                # Compute channel-wise variance
                channel_vars = np.var(segment, axis=1)
                
                # Skip if any channel is dead (variance too low)
                if np.any(channel_vars < min_variance):
                    continue
                
                # Skip if any channel has excessive artifacts (variance too high)
                if np.any(channel_vars > max_variance):
                    continue
                
                # Compute overall artifact score (lower is better)
                artifact_score = np.mean(channel_vars)
                
                if artifact_score < min_artifact_score:
                    min_artifact_score = artifact_score
                    best_segment = segment
            
            if best_segment is not None:
                logger.info(f"Found best resting segment with artifact score: {min_artifact_score:.2e}")
                return best_segment.astype(np.float64)
            else:
                # Fallback: return first 60 seconds
                logger.warning("No artifact-free segment found. Returning first 60 seconds.")
                return ica_data[:, :n_samples].astype(np.float64)
            
        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise
            raise PreprocessingError(f"Resting segment extraction failed: {str(e)}")
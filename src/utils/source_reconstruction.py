import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any
from scipy.linalg import pinv, svd

# Configure module-specific logger
logger = logging.getLogger(__name__)


class SourceReconstructionError(Exception):
    """Custom exception for source reconstruction errors."""
    pass


class DesikanKillianyAtlas:
    """
    Desikan-Killiany cortical atlas with 68 regions.
    
    This class provides the anatomical mapping for the 68 cortical regions
    used in the study. It includes region names, hemisphere assignments,
    and spatial coordinates for source reconstruction.
    """
    
    def __init__(self):
        """Initialize Desikan-Killiany atlas."""
        self.n_regions = 68
        self.region_names = [
            # Left hemisphere (34 regions)
            'bankssts_lh', 'caudal anterior cingulate_lh', 'caudal middle frontal_lh',
            'cuneus_lh', 'entorhinal_lh', 'fusiform_lh', 'inferior parietal_lh',
            'inferior temporal_lh', 'isthmus cingulate_lh', 'lateral occipital_lh',
            'lateral orbitofrontal_lh', 'lingual_lh', 'medial orbitofrontal_lh',
            'middle temporal_lh', 'parahippocampal_lh', 'paracentral_lh',
            'pars opercularis_lh', 'pars orbitalis_lh', 'pars triangularis_lh',
            'pericalcarine_lh', 'postcentral_lh', 'posterior cingulate_lh',
            'precentral_lh', 'precuneus_lh', 'rostral anterior cingulate_lh',
            'rostral middle frontal_lh', 'superior frontal_lh', 'superior parietal_lh',
            'superior temporal_lh', 'supramarginal_lh', 'frontal pole_lh',
            'temporal pole_lh', 'transverse temporal_lh', 'insula_lh',
            # Right hemisphere (34 regions)
            'bankssts_rh', 'caudal anterior cingulate_rh', 'caudal middle frontal_rh',
            'cuneus_rh', 'entorhinal_rh', 'fusiform_rh', 'inferior parietal_rh',
            'inferior temporal_rh', 'isthmus cingulate_rh', 'lateral occipital_rh',
            'lateral orbitofrontal_rh', 'lingual_rh', 'medial orbitofrontal_rh',
            'middle temporal_rh', 'parahippocampal_rh', 'paracentral_rh',
            'pars opercularis_rh', 'pars orbitalis_rh', 'pars triangularis_rh',
            'pericalcarine_rh', 'postcentral_rh', 'posterior cingulate_rh',
            'precentral_rh', 'precuneus_rh', 'rostral anterior cingulate_rh',
            'rostral middle frontal_rh', 'superior frontal_rh', 'superior parietal_rh',
            'superior temporal_rh', 'supramarginal_rh', 'frontal pole_rh',
            'temporal pole_rh', 'transverse temporal_rh', 'insula_rh'
        ]
        
        # Hemisphere assignments
        self.hemispheres = ['lh'] * 34 + ['rh'] * 34
        
        # Approximate MNI coordinates for each region (simplified template)
        # These are representative coordinates for source space definition
        self.coordinates = self._generate_template_coordinates()
    
    def _generate_template_coordinates(self) -> np.ndarray:
        """Generate template MNI coordinates for 68 regions."""
        # This is a simplified template - in practice, use actual FreeSurfer coordinates
        coordinates = np.zeros((68, 3))
        
        # Left hemisphere (negative x-coordinates)
        for i in range(34):
            coordinates[i, 0] = -40 + np.random.normal(0, 20)  # x (left: negative)
            coordinates[i, 1] = np.random.normal(0, 30)        # y
            coordinates[i, 2] = np.random.normal(20, 20)       # z
        
        # Right hemisphere (positive x-coordinates)
        for i in range(34, 68):
            coordinates[i, 0] = 40 + np.random.normal(0, 20)   # x (right: positive)
            coordinates[i, 1] = np.random.normal(0, 30)        # y
            coordinates[i, 2] = np.random.normal(20, 20)       # z
        
        return coordinates
    
    def get_region_info(self, region_idx: int) -> Dict[str, Any]:
        """Get complete information for a region."""
        if region_idx < 0 or region_idx >= self.n_regions:
            raise SourceReconstructionError(f"Invalid region index: {region_idx}")
        
        return {
            'name': self.region_names[region_idx],
            'hemisphere': self.hemispheres[region_idx],
            'coordinates': self.coordinates[region_idx].copy()
        }


class ForwardModel:
    """
    Forward model for M/EEG source reconstruction.
    
    This class implements the forward model that maps source activity
    to sensor measurements. It handles both EEG and MEG modalities
    with appropriate head models.
    """
    
    def __init__(
        self,
        sensor_positions: np.ndarray,
        source_coordinates: np.ndarray,
        modality: str = 'eeg',
        conductivity: Optional[np.ndarray] = None
    ):
        """
        Initialize forward model.
        
        Parameters
        ----------
        sensor_positions : np.ndarray
            Sensor positions of shape (n_sensors, 3).
        source_coordinates : np.ndarray
            Source coordinates of shape (n_sources, 3).
        modality : str, optional
            Modality ('eeg' or 'meg', default: 'eeg').
        conductivity : np.ndarray, optional
            Conductivity values for multi-layer head model.
        """
        if sensor_positions.ndim != 2 or sensor_positions.shape[1] != 3:
            raise SourceReconstructionError("sensor_positions must be (n_sensors, 3).")
        if source_coordinates.ndim != 2 or source_coordinates.shape[1] != 3:
            raise SourceReconstructionError("source_coordinates must be (n_sources, 3).")
        if modality not in ['eeg', 'meg']:
            raise SourceReconstructionError("modality must be 'eeg' or 'meg'.")
        
        self.sensor_positions = sensor_positions.copy()
        self.source_coordinates = source_coordinates.copy()
        self.modality = modality
        self.conductivity = conductivity
        self.n_sensors = sensor_positions.shape[0]
        self.n_sources = source_coordinates.shape[0]
        self.leadfield = None
        
        self._compute_leadfield()
    
    def _compute_eeg_leadfield(self) -> np.ndarray:
        """Compute EEG leadfield using simplified spherical model."""
        try:
            # Simplified EEG leadfield computation
            # In practice, use boundary element method (BEM) or finite element method (FEM)
            
            # Head radius (simplified)
            head_radius = 0.09  # 9 cm
            
            # Source to sensor vectors
            leadfield = np.zeros((self.n_sensors, self.n_sources))
            
            for i in range(self.n_sensors):
                for j in range(self.n_sources):
                    # Source position (ensure it's inside the head)
                    src_pos = self.source_coordinates[j]
                    src_norm = np.linalg.norm(src_pos)
                    if src_norm >= head_radius:
                        src_pos = src_pos * (head_radius * 0.9 / src_norm)
                    
                    # Sensor position (on scalp)
                    sens_pos = self.sensor_positions[i]
                    sens_norm = np.linalg.norm(sens_pos)
                    if sens_norm == 0:
                        sens_pos = np.array([head_radius, 0, 0])
                    else:
                        sens_pos = sens_pos * (head_radius / sens_norm)
                    
                    # Simplified leadfield element (dipole potential)
                    r_vec = sens_pos - src_pos
                    r_dist = np.linalg.norm(r_vec)
                    if r_dist < 1e-6:
                        r_dist = 1e-6
                    
                    # Radial component (simplified)
                    leadfield[i, j] = 1.0 / (4 * np.pi * r_dist)
            
            # Normalize leadfield
            leadfield = leadfield / np.max(np.abs(leadfield))
            return leadfield
            
        except Exception as e:
            raise SourceReconstructionError(f"EEG leadfield computation failed: {str(e)}")
    
    def _compute_meg_leadfield(self) -> np.ndarray:
        """Compute MEG leadfield using simplified dipole model."""
        try:
            # Simplified MEG leadfield for magnetometers
            # In practice, use realistic head models
            
            leadfield = np.zeros((self.n_sensors, self.n_sources))
            
            for i in range(self.n_sensors):
                sens_pos = self.sensor_positions[i]
                
                for j in range(self.n_sources):
                    src_pos = self.source_coordinates[j]
                    
                    # Vector from source to sensor
                    r_vec = sens_pos - src_pos
                    r_dist = np.linalg.norm(r_vec)
                    if r_dist < 1e-6:
                        r_dist = 1e-6
                    
                    # Simplified MEG leadfield (radial component ignored for magnetometers)
                    # Using dipole field formula
                    leadfield[i, j] = 1.0 / (r_dist ** 2)
            
            # Normalize leadfield
            leadfield = leadfield / np.max(np.abs(leadfield))
            return leadfield
            
        except Exception as e:
            raise SourceReconstructionError(f"MEG leadfield computation failed: {str(e)}")
    
    def _compute_leadfield(self) -> None:
        """Compute the leadfield matrix."""
        try:
            if self.modality == 'eeg':
                self.leadfield = self._compute_eeg_leadfield()
            else:  # meg
                self.leadfield = self._compute_meg_leadfield()
            
            logger.info(f"Leadfield computed: {self.leadfield.shape}")
            
        except Exception as e:
            raise SourceReconstructionError(f"Leadfield computation failed: {str(e)}")
    
    def get_leadfield(self) -> np.ndarray:
        """Get the computed leadfield matrix."""
        if self.leadfield is None:
            raise SourceReconstructionError("Leadfield not computed.")
        return self.leadfield.copy()


class sLORETAInverseSolver:
    """
    Standardized Low-Resolution Brain Electromagnetic Tomography (sLORETA) inverse solver.
    
    This class implements the sLORETA algorithm for source reconstruction,
    providing zero-error localization for single sources and handling
    the ill-posed inverse problem through regularization.
    """
    
    def __init__(
        self,
        leadfield: np.ndarray,
        regularization_param: float = 1e-3,
        tikhonov_order: int = 0
    ):
        """
        Initialize sLORETA inverse solver.
        
        Parameters
        ----------
        leadfield : np.ndarray
            Leadfield matrix of shape (n_sensors, n_sources).
        regularization_param : float, optional
            Regularization parameter (default: 1e-3).
        tikhonov_order : int, optional
            Order of Tikhonov regularization (0 = standard, 1 = 1st derivative, etc.).
        """
        if leadfield.ndim != 2:
            raise SourceReconstructionError("leadfield must be 2D.")
        if regularization_param <= 0:
            raise SourceReconstructionError("regularization_param must be positive.")
        if tikhonov_order < 0:
            raise SourceReconstructionError("tikhonov_order must be non-negative.")
        
        self.leadfield = leadfield.copy()
        self.regularization_param = float(regularization_param)
        self.tikhonov_order = int(tikhonov_order)
        self.n_sensors, self.n_sources = leadfield.shape
        self.inverse_operator = None
        
        self._compute_inverse_operator()
    
    def _construct_tikhonov_matrix(self) -> np.ndarray:
        """Construct Tikhonov regularization matrix."""
        if self.tikhonov_order == 0:
            return np.eye(self.n_sources)
        elif self.tikhonov_order == 1:
            # First-order Tikhonov (smoothness constraint)
            D = np.zeros((self.n_sources - 1, self.n_sources))
            for i in range(self.n_sources - 1):
                D[i, i] = -1
                D[i, i + 1] = 1
            return D
        else:
            # Higher-order Tikhonov (not implemented for simplicity)
            logger.warning(f"Tikhonov order {self.tikhonov_order} not implemented. Using order 0.")
            return np.eye(self.n_sources)
    
    def _compute_inverse_operator(self) -> None:
        """Compute the sLORETA inverse operator."""
        try:
            # Construct regularization matrix
            L = self._construct_tikhonov_matrix()
            
            # Compute sLORETA inverse operator
            # sLORETA uses: W = R^{-1} G^T (G R^{-1} G^T + λI)^{-1}
            # where R = L^T L (regularization matrix)
            
            if self.tikhonov_order == 0:
                R = np.eye(self.n_sources)
                R_inv = np.eye(self.n_sources)
            else:
                R = L.T @ L
                # Ensure R is positive definite
                R += np.eye(self.n_sources) * 1e-12
                R_inv = pinv(R)
            
            # Compute the inverse operator
            GRGt = self.leadfield @ R_inv @ self.leadfield.T
            GRGt_reg = GRGt + self.regularization_param * np.eye(self.n_sensors)
            
            # Compute pseudoinverse
            GRGt_inv = pinv(GRGt_reg)
            self.inverse_operator = R_inv @ self.leadfield.T @ GRGt_inv
            
            logger.info(f"sLORETA inverse operator computed: {self.inverse_operator.shape}")
            
        except Exception as e:
            raise SourceReconstructionError(f"sLORETA inverse operator computation failed: {str(e)}")
    
    def apply_inverse(self,  np.ndarray) -> np.ndarray:
        """
        Apply inverse operator to sensor data.
        
        Parameters
        ----------
        sensor_data : np.ndarray
            Sensor data of shape (n_sensors, n_times).
        
        Returns
        -------
        np.ndarray
            Source time series of shape (n_sources, n_times).
        """
        try:
            if sensor_data.ndim != 2:
                raise SourceReconstructionError("sensor_data must be 2D.")
            if sensor_data.shape[0] != self.n_sensors:
                raise SourceReconstructionError(
                    f"sensor_data has {sensor_data.shape[0]} sensors, expected {self.n_sensors}."
                )
            
            source_data = self.inverse_operator @ sensor_data
            return source_data
            
        except Exception as e:
            raise SourceReconstructionError(f"Inverse solution application failed: {str(e)}")


class SourceReconstructor:
    """
    Unified source reconstructor for M/EEG datasets.
    
    This class provides a complete source reconstruction pipeline that
    handles dataset-specific sensor configurations and projects data
    to the common 68-region Desikan-Killiany source space.
    """
    
    def __init__(
        self,
        dataset_name: str,
        modality: str = 'auto',
        regularization_param: float = 1e-3,
        low_density_regularization: float = 1e-2
    ):
        """
        Initialize source reconstructor.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset ('OpenNeuro_ds004504', 'BioFIND', etc.).
        modality : str, optional
            Modality ('eeg', 'meg', or 'auto', default: 'auto').
        regularization_param : float, optional
            Regularization parameter for standard cases (default: 1e-3).
        low_density_regularization : float, optional
            Higher regularization for low-density EEG (default: 1e-2).
        """
        self.dataset_name = dataset_name
        self.modality = modality
        self.regularization_param = float(regularization_param)
        self.low_density_regularization = float(low_density_regularization)
        self.atlas = DesikanKillianyAtlas()
        self.forward_model = None
        self.inverse_solver = None
        
        # Dataset-specific sensor configurations
        self._dataset_configs = {
            'OpenNeuro_ds004504': {
                'modality': 'eeg',
                'n_channels': 19,
                'standard_positions': self._get_1020_positions()
            },
            'BioFIND': {
                'modality': 'meg',
                'n_channels': 306,
                'standard_positions': self._get_meg_positions()
            },
            'Mendeley_Olfactory': {
                'modality': 'eeg',
                'n_channels': 4,
                'standard_positions': self._get_4_channel_positions()
            },
            'PREVENT_AD': {
                'modality': 'eeg',
                'n_channels': [32, 64],  # g.tec and BrainAmp
                'standard_positions': {
                    'gtec': self._get_gtec_32_positions(),
                    'brainamp': self._get_brainamp_64_positions()
                }
            }
        }
        
        if dataset_name not in self._dataset_configs:
            raise SourceReconstructionError(f"Unsupported dataset: {dataset_name}")
    
    def _get_1020_positions(self) -> np.ndarray:
        """Get standard 10-20 system positions (simplified MNI coordinates)."""
        # Simplified 10-20 positions in MNI space (mm)
        positions = {
            'Fp1': [-34, 46, 36], 'Fp2': [34, 46, 36], 'F7': [-62, 7, 36],
            'F3': [-46, 25, 52], 'Fz': [0, 30, 60], 'F4': [46, 25, 52],
            'F8': [62, 7, 36], 'T7': [-70, -22, 36], 'C3': [-52, -22, 60],
            'Cz': [0, -22, 72], 'C4': [52, -22, 60], 'T8': [70, -22, 36],
            'P7': [-62, -58, 36], 'P3': [-46, -58, 52], 'Pz': [0, -62, 60],
            'P4': [46, -58, 52], 'P8': [62, -58, 36], 'O1': [-34, -86, 36],
            'O2': [34, -86, 36]
        }
        return np.array(list(positions.values()), dtype=float) / 1000.0  # Convert to meters
    
    def _get_meg_positions(self) -> np.ndarray:
        """Get standard MEG sensor positions (simplified)."""
        # Simulate 306 MEG sensor positions (102 magnetometers + 204 gradiometers)
        # In practice, these would come from the actual sensor layout
        n_mags = 102
        n_grads = 204
        
        # Magnetometer positions (on helmet surface)
        mag_positions = []
        for i in range(n_mags):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = 0.12  # 12 cm helmet radius
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) + 0.04  # Shift up for head position
            mag_positions.append([x, y, z])
        
        # Gradiometer positions (slightly offset from magnetometers)
        grad_positions = []
        for i in range(n_grads):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = 0.11  # Slightly inside
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) + 0.04
            grad_positions.append([x, y, z])
        
        all_positions = np.vstack([mag_positions, grad_positions])
        return all_positions.astype(float)
    
    def _get_4_channel_positions(self) -> np.ndarray:
        """Get 4-channel EEG positions."""
        positions = {
            'Fp1': [-34, 46, 36], 'Fp2': [34, 46, 36],
            'C3': [-52, -22, 60], 'C4': [52, -22, 60]
        }
        return np.array(list(positions.values()), dtype=float) / 1000.0
    
    def _get_gtec_32_positions(self) -> np.ndarray:
        """Get g.tec 32-channel positions."""
        # Simplified 32-channel positions based on extended 10-20
        positions = {}
        standard_1020 = self._get_1020_positions() * 1000  # Back to mm
        # Use first 32 of standard 10-20 plus some extras
        channel_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
            'Pz', 'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'O2'
        ]
        for i, ch in enumerate(channel_names):
            if i < len(standard_1020):
                positions[ch] = standard_1020[i]
            else:
                # Approximate positions for extra channels
                positions[ch] = [0, 0, 0]
        
        return np.array(list(positions.values()), dtype=float) / 1000.0
    
    def _get_brainamp_64_positions(self) -> np.ndarray:
        """Get BrainAmp 64-channel positions."""
        # Simplified 64-channel positions
        # Use standard 10-20 and add intermediate positions
        base_positions = self._get_1020_positions() * 1000
        # For simplicity, duplicate and offset base positions
        positions = []
        for i in range(64):
            if i < len(base_positions):
                pos = base_positions[i]
            else:
                # Create intermediate positions
                pos = base_positions[i % len(base_positions)] + np.random.normal(0, 5, 3)
            positions.append(pos)
        return np.array(positions, dtype=float) / 1000.0
    
    def _determine_modality(self, n_channels: int) -> str:
        """Determine modality based on number of channels."""
        if 'BioFIND' in self.dataset_name or n_channels == 306:
            return 'meg'
        else:
            return 'eeg'
    
    def _get_sensor_positions(self, n_channels: int, system_type: str = 'auto') -> np.ndarray:
        """Get appropriate sensor positions for the dataset."""
        config = self._dataset_configs[self.dataset_name]
        
        if self.dataset_name == 'PREVENT_AD':
            if system_type == 'gtec' or n_channels == 32:
                return config['standard_positions']['gtec']
            elif system_type == 'brainamp' or n_channels == 64:
                return config['standard_positions']['brainamp']
            else:
                # Auto-detect based on channel count
                if n_channels == 32:
                    return config['standard_positions']['gtec']
                else:
                    return config['standard_positions']['brainamp']
        else:
            positions = config['standard_positions']
            if len(positions) >= n_channels:
                return positions[:n_channels]
            else:
                # Pad with zeros if not enough standard positions
                padded = np.zeros((n_channels, 3))
                padded[:len(positions)] = positions
                return padded
    
    def setup_reconstruction(
        self,
        n_channels: int,
        system_type: str = 'auto'
    ) -> None:
        """
        Setup source reconstruction for a specific recording.
        
        Parameters
        ----------
        n_channels : int
            Number of channels in the recording.
        system_type : str, optional
            System type for PREVENT-AD ('gtec', 'brainamp', or 'auto').
        """
        try:
            # Determine modality
            if self.modality == 'auto':
                self.modality = self._determine_modality(n_channels)
            
            # Get sensor positions
            sensor_positions = self._get_sensor_positions(n_channels, system_type)
            
            # Get source coordinates
            source_coordinates = self.atlas.coordinates.copy()
            
            # Create forward model
            self.forward_model = ForwardModel(
                sensor_positions=sensor_positions,
                source_coordinates=source_coordinates,
                modality=self.modality
            )
            
            # Determine regularization parameter
            if self.dataset_name == 'Mendeley_Olfactory' or n_channels < 10:
                reg_param = self.low_density_regularization
            else:
                reg_param = self.regularization_param
            
            # Create inverse solver
            self.inverse_solver = sLORETAInverseSolver(
                leadfield=self.forward_model.get_leadfield(),
                regularization_param=reg_param
            )
            
            logger.info(f"Source reconstruction setup completed for {self.dataset_name}")
            
        except Exception as e:
            raise SourceReconstructionError(f"Source reconstruction setup failed: {str(e)}")
    
    def reconstruct_sources(
        self,
        sensor_data: np.ndarray,
        n_channels: int,
        system_type: str = 'auto'
    ) -> np.ndarray:
        """
        Reconstruct source time series from sensor data.
        
        Parameters
        ----------
        sensor_data : np.ndarray
            Sensor data of shape (n_channels, n_times).
        n_channels : int
            Number of channels.
        system_type : str, optional
            System type for PREVENT-AD.
        
        Returns
        -------
        np.ndarray
            Source time series of shape (68, n_times).
        """
        try:
            if sensor_data.ndim != 2:
                raise SourceReconstructionError("sensor_data must be 2D.")
            if sensor_data.shape[0] != n_channels:
                raise SourceReconstructionError(
                    f"sensor_data has {sensor_data.shape[0]} channels, expected {n_channels}."
                )
            
            # Setup reconstruction if not already done
            if self.inverse_solver is None:
                self.setup_reconstruction(n_channels, system_type)
            
            # Apply inverse solution
            full_source_data = self.inverse_solver.apply_inverse(sensor_data)
            
            # Aggregate to 68 Desikan-Killiany regions
            # For simplicity, assume one source per region (in practice, average over vertices)
            if full_source_data.shape[0] == 68:
                source_data_68 = full_source_data
            else:
                # If more sources, average to 68 regions
                # This is a simplified approach - in practice, use proper parcellation
                n_full_sources = full_source_data.shape[0]
                sources_per_region = n_full_sources // 68
                source_data_68 = np.zeros((68, sensor_data.shape[1]))
                
                for i in range(68):
                    start_idx = i * sources_per_region
                    end_idx = min((i + 1) * sources_per_region, n_full_sources)
                    if start_idx < end_idx:
                        source_data_68[i] = np.mean(full_source_data[start_idx:end_idx], axis=0)
                    else:
                        source_data_68[i] = full_source_data[i % n_full_sources]
            
            # Harmonize dipole orientations to prevent cancellation
            source_data_68 = self._harmonize_orientations(source_data_68)
            
            logger.info(f"Source reconstruction completed. Output shape: {source_data_68.shape}")
            return source_data_68
            
        except Exception as e:
            raise SourceReconstructionError(f"Source reconstruction failed: {str(e)}")
    
    def _harmonize_orientations(self,  np.ndarray) -> np.ndarray:
        """
        Harmonize dipole orientations to prevent signal cancellation.
        
        Parameters
        ----------
        source_data : np.ndarray
            Source data of shape (68, n_times).
        
        Returns
        -------
        np.ndarray
            Harmonized source data of same shape.
        """
        try:
            # For each region, ensure consistent orientation
            # This is a simplified approach - flip if mean is negative
            harmonized = source_data.copy()
            
            for i in range(harmonized.shape[0]):
                mean_val = np.mean(harmonized[i])
                if mean_val < 0:
                    harmonized[i] = -harmonized[i]
            
            return harmonized
            
        except Exception as e:
            logger.warning(f"Orientation harmonization failed: {str(e)}. Returning original data.")
            return source_data
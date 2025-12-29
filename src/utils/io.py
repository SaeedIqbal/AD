import os
import json
import yaml
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Configure module-specific logger
logger = logging.getLogger(__name__)


class IOError(Exception):
    """Custom exception for I/O operations."""
    pass


class DatasetConfig:
    """
    Dataset configuration manager for the four datasets in the study.
    
    This class handles loading, validation, and access to dataset-specific 
    configuration parameters including file paths, preprocessing parameters,
    and metadata. It supports the OpenNeuro ds004504, BioFIND, Mendeley Olfactory, 
    and PREVENT-AD datasets.
    """
    
    SUPPORTED_DATASETS = {
        'OpenNeuro_ds004504': {
            'required_files': ['.set', '.edf', '.mat'],
            'channels': 19,
            'default_sfreq': 500.0,
            'montage': '10-20'
        },
        'BioFIND': {
            'required_files': ['.fif'],
            'channels': 306,
            'default_sfreq': 1000.0,
            'montage': 'MEG'
        },
        'Mendeley_Olfactory': {
            'required_files': ['.mat', '.csv'],
            'channels': 4,
            'default_sfreq': 256.0,
            'montage': '4-channel'
        },
        'PREVENT_AD': {
            'required_files': ['.vhdr', '.gdf', '.edf', '.bdf'],
            'channels': [32, 64],  # g.tec and BrainAmp
            'default_sfreq': [256.0, 500.0],
            'montage': 'extended-10-20'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize dataset configuration manager.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file. If None, uses default structure.
        
        Raises
        ------
        IOError
            If configuration loading fails.
        """
        self.config_path = config_path
        self.dataset_configs = {}
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for datasets."""
        return {
            'datasets': {
                'OpenNeuro_ds004504': {
                    'path': '/home/phd/datasets/OpenNeuro_ds004504',
                    'type': 'eeg',
                    'format': 'eeglab'
                },
                'BioFIND': {
                    'path': '/home/phd/datasets/BioFIND',
                    'type': 'meg',
                    'format': 'fif'
                },
                'Mendeley_Olfactory': {
                    'path': '/home/phd/datasets/Mendeley_Olfactory',
                    'type': 'eeg',
                    'format': 'mat'
                },
                'PREVENT_AD': {
                    'path': '/home/phd/datasets/PREVENT_AD',
                    'type': 'eeg',
                    'format': 'brainstorm'
                }
            }
        }
    
    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        self.dataset_configs = yaml.safe_load(f)
                    elif self.config_path.endswith('.json'):
                        self.dataset_configs = json.load(f)
                    else:
                        raise IOError("Unsupported config format. Use .yaml, .yml, or .json.")
            else:
                self.dataset_configs = self._load_default_config()
            
            self._validate_config()
            logger.info("Dataset configuration loaded successfully.")
            
        except Exception as e:
            raise IOError(f"Failed to load dataset configuration: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate loaded configuration."""
        if 'datasets' not in self.dataset_configs:
            raise IOError("Configuration must contain 'datasets' key.")
        
        for dataset_name, config in self.dataset_configs['datasets'].items():
            if 'path' not in config:
                raise IOError(f"Dataset {dataset_name} missing 'path' in configuration.")
            if not os.path.exists(config['path']):
                logger.warning(f"Dataset path does not exist: {config['path']}")
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """
        Get dataset path for a given dataset name.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        
        Returns
        -------
        str
            Path to the dataset.
        
        Raises
        ------
        IOError
            If dataset is not configured.
        """
        if dataset_name not in self.dataset_configs['datasets']:
            raise IOError(f"Dataset {dataset_name} not found in configuration.")
        return self.dataset_configs['datasets'][dataset_name]['path']
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get complete dataset information.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        
        Returns
        -------
        dict
            Dataset information.
        """
        if dataset_name not in self.dataset_configs['datasets']:
            raise IOError(f"Dataset {dataset_name} not found in configuration.")
        return self.dataset_configs['datasets'][dataset_name]
    
    def list_datasets(self) -> List[str]:
        """List all configured datasets."""
        return list(self.dataset_configs['datasets'].keys())


class DataSaver:
    """
    Unified data saver for processed results and intermediate outputs.
    
    This class handles saving of preprocessed data, IAC trajectories, 
    recurrence metrics, and other analysis results in standardized formats
    suitable for reproducibility and sharing.
    """
    
    def __init__(self, base_output_dir: str = '/home/phd/results'):
        """
        Initialize data saver.
        
        Parameters
        ----------
        base_output_dir : str, optional
            Base directory for output files (default: '/home/phd/results').
        
        Raises
        ------
        IOError
            If output directory cannot be created.
        """
        self.base_output_dir = base_output_dir
        self._ensure_directory_structure()
    
    def _ensure_directory_structure(self) -> None:
        """Create required directory structure."""
        try:
            directories = [
                'preprocessed',
                'iac_trajectories', 
                'metrics',
                'figures',
                'surrogates',
                'meta_states'
            ]
            
            for dir_name in directories:
                dir_path = os.path.join(self.base_output_dir, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            logger.info(f"Output directory structure created at {self.base_output_dir}")
            
        except Exception as e:
            raise IOError(f"Failed to create output directory structure: {str(e)}")
    
    def _validate_data(self, data: Any, name: str) -> None:
        """Validate data before saving."""
        if data is None:
            raise IOError(f"Cannot save None data for {name}.")
        
        if isinstance(data, np.ndarray) and not np.all(np.isfinite(data)):
            raise IOError(f"Data for {name} contains non-finite values.")
    
    def save_preprocessed_data(
        self,
        dataset_name: str,
        subject_id: str,
        data: np.ndarray,
        sfreq: float,
        ch_names: List[str]
    ) -> str:
        """
        Save preprocessed EEG/MEG data.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        subject_id : str
            Subject identifier.
        data : np.ndarray
            Preprocessed data of shape (n_channels, n_times).
        sfreq : float
            Sampling frequency in Hz.
        ch_names : list of str
            Channel names.
        
        Returns
        -------
        str
            Path to saved file.
        """
        try:
            self._validate_data(data, f"preprocessed_{dataset_name}_{subject_id}")
            
            filename = f"{dataset_name}_{subject_id}_preprocessed.npz"
            filepath = os.path.join(self.base_output_dir, 'preprocessed', filename)
            
            np.savez_compressed(
                filepath,
                data=data,
                sfreq=sfreq,
                ch_names=ch_names,
                dataset_name=dataset_name,
                subject_id=subject_id
            )
            
            logger.info(f"Preprocessed data saved: {filepath}")
            return filepath
            
        except Exception as e:
            raise IOError(f"Failed to save preprocessed data: {str(e)}")
    
    def save_iac_trajectory(
        self,
        dataset_name: str,
        subject_id: str,
        C_traj: np.ndarray,
        band: Tuple[float, float],
        sfreq: float
    ) -> str:
        """
        Save IAC trajectory.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        subject_id : str
            Subject identifier.
        C_traj : np.ndarray
            IAC trajectory of shape (n_times, n_rois, n_rois).
        band : tuple of (float, float)
            Frequency band (low, high) in Hz.
        sfreq : float
            Sampling frequency in Hz.
        
        Returns
        -------
        str
            Path to saved file.
        """
        try:
            self._validate_data(C_traj, f"iac_{dataset_name}_{subject_id}")
            
            # Create safe band string for filename
            band_str = f"{band[0]:.1f}-{band[1]:.1f}Hz".replace('.', '_')
            filename = f"{dataset_name}_{subject_id}_iac_{band_str}.npz"
            filepath = os.path import os.path.join(self.base_output_dir, 'iac_trajectories', filename)
            
            np.savez_compressed(
                filepath,
                C_traj=C_traj,
                band=band,
                sfreq=sfreq,
                dataset_name=dataset_name,
                subject_id=subject_id
            )
            
            logger.info(f"IAC trajectory saved: {filepath}")
            return filepath
            
        except Exception as e:
            raise IOError(f"Failed to save IAC trajectory: {str(e)}")
    
    def save_metrics(
        self,
        dataset_name: str,
        subject_id: str,
        metrics: Dict[str, Union[float, np.ndarray]]
    ) -> str:
        """
        Save computed metrics.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        subject_id : str
            Subject identifier.
        metrics : dict
            Dictionary of metric names and values.
        
        Returns
        -------
        str
            Path to saved file.
        """
        try:
            filename = f"{dataset_name}_{subject_id}_metrics.json"
            filepath = os.path.join(self.base_output_dir, 'metrics', filename)
            
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_metrics[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_metrics[key] = int(value)
                else:
                    serializable_metrics[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Metrics saved: {filepath}")
            return filepath
            
        except Exception as e:
            raise IOError(f"Failed to save metrics: {str(e)}")
    
    def save_meta_states(
        self,
        dataset_name: str,
        meta_state_centroids: np.ndarray,
        meta_state_covariances: Optional[np.ndarray] = None
    ) -> str:
        """
        Save group-level meta-state definitions.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        meta_state_centroids : np.ndarray
            Meta-state centroids.
        meta_state_covariances : np.ndarray, optional
            Meta-state covariances.
        
        Returns
        -------
        str
            Path to saved file.
        """
        try:
            self._validate_data(meta_state_centroids, f"meta_states_{dataset_name}")
            
            filename = f"{dataset_name}_meta_states.npz"
            filepath = os.path.join(self.base_output_dir, 'meta_states', filename)
            
            save_dict = {'centroids': meta_state_centroids}
            if meta_state_covariances is not None:
                save_dict['covariances'] = meta_state_covariances
            
            np.savez_compressed(filepath, **save_dict)
            
            logger.info(f"Meta-states saved: {filepath}")
            return filepath
            
        except Exception as e:
            raise IOError(f"Failed to save meta-states: {str(e)}")


class DataLoader:
    """
    Unified data loader for processed results and intermediate outputs.
    
    This class handles loading of preprocessed data, IAC trajectories,
    metrics, and other analysis results in a standardized way.
    """
    
    def __init__(self, base_input_dir: str = '/home/phd/results'):
        """
        Initialize data loader.
        
        Parameters
        ----------
        base_input_dir : str, optional
            Base directory for input files (default: '/home/phd/results').
        """
        self.base_input_dir = base_input_dir
    
    def load_preprocessed_data(self, filepath: str) -> Dict[str, Any]:
        """
        Load preprocessed EEG/MEG data.
        
        Parameters
        ----------
        filepath : str
            Path to preprocessed data file.
        
        Returns
        -------
        dict
            Dictionary containing data, sfreq, ch_names, etc.
        """
        try:
            if not os.path.exists(filepath):
                raise IOError(f"File not found: {filepath}")
            
            data = np.load(filepath)
            result = {key: data[key] for key in data.keys()}
            logger.info(f"Preprocessed data loaded: {filepath}")
            return result
            
        except Exception as e:
            raise IOError(f"Failed to load preprocessed data: {str(e)}")
    
    def load_iac_trajectory(self, filepath: str) -> Dict[str, Any]:
        """
        Load IAC trajectory.
        
        Parameters
        ----------
        filepath : str
            Path to IAC trajectory file.
        
        Returns
        -------
        dict
            Dictionary containing C_traj, band, sfreq, etc.
        """
        try:
            if not os.path.exists(filepath):
                raise IOError(f"File not found: {filepath}")
            
            data = np.load(filepath)
            result = {key: data[key] for key in data.keys()}
            logger.info(f"IAC trajectory loaded: {filepath}")
            return result
            
        except Exception as e:
            raise IOError(f"Failed to load IAC trajectory: {str(e)}")
    
    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Load computed metrics.
        
        Parameters
        ----------
        filepath : str
            Path to metrics file.
        
        Returns
        -------
        dict
            Dictionary of metric names and values.
        """
        try:
            if not os.path.exists(filepath):
                raise IOError(f"File not found: {filepath}")
            
            with open(filepath, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Metrics loaded: {filepath}")
            return metrics
            
        except Exception as e:
            raise IOError(f"Failed to load metrics: {str(e)}")
    
    def load_meta_states(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load group-level meta-state definitions.
        
        Parameters
        ----------
        filepath : str
            Path to meta-states file.
        
        Returns
        -------
        dict
            Dictionary containing centroids and covariances.
        """
        try:
            if not os.path.exists(filepath):
                raise IOError(f"File not found: {filepath}")
            
            data = np.load(filepath)
            result = {key: data[key] for key in data.keys()}
            logger.info(f"Meta-states loaded: {filepath}")
            return result
            
        except Exception as e:
            raise IOError(f"Failed to load meta-states: {str(e)}")
    
    def find_subject_files(
        self,
        dataset_name: str,
        subject_id: str,
        file_type: str = 'preprocessed'
    ) -> List[str]:
        """
        Find all files for a given subject and file type.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        subject_id : str
            Subject identifier.
        file_type : str, optional
            Type of files to find (default: 'preprocessed').
        
        Returns
        -------
        list of str
            List of matching file paths.
        """
        try:
            search_dir = os.path.join(self.base_input_dir, file_type)
            if not os.path.exists(search_dir):
                return []
            
            pattern = f"{dataset_name}_{subject_id}_*"
            matching_files = []
            
            for filename in os.listdir(search_dir):
                if filename.startswith(f"{dataset_name}_{subject_id}_"):
                    matching_files.append(os.path.join(search_dir, filename))
            
            return sorted(matching_files)
            
        except Exception as e:
            logger.warning(f"Failed to find subject files: {str(e)}")
            return []


class ResultsOrganizer:
    """
    Results organizer for systematic output management.
    
    This class provides utilities for organizing results by dataset,
    clinical group, and analysis type, facilitating systematic evaluation
    and reporting.
    """
    
    def __init__(self, results_dir: str = '/home/phd/results'):
        """
        Initialize results organizer.
        
        Parameters
        ----------
        results_dir : str, optional
            Base results directory (default: '/home/phd/results').
        """
        self.results_dir = results_dir
        self.data_loader = DataLoader(results_dir)
    
    def organize_by_clinical_group(
        self,
        dataset_name: str,
        clinical_groups: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Organize files by clinical group.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        clinical_groups : dict
            Dictionary mapping group names to subject IDs.
        
        Returns
        -------
        dict
            Organized file structure by group and file type.
        """
        try:
            organized = {}
            file_types = ['metrics', 'iac_trajectories', 'preprocessed']
            
            for group_name, subject_ids in clinical_groups.items():
                organized[group_name] = {}
                for file_type in file_types:
                    organized[group_name][file_type] = []
                    for subject_id in subject_ids:
                        files = self.data_loader.find_subject_files(
                            dataset_name, subject_id, file_type
                        )
                        organized[group_name][file_type].extend(files)
            
            return organized
            
        except Exception as e:
            raise IOError(f"Failed to organize results by clinical group: {str(e)}")
    
    def create_results_summary(
        self,
        dataset_name: str,
        output_filename: str = 'results_summary.json'
    ) -> str:
        """
        Create a comprehensive results summary.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        output_filename : str, optional
            Output filename (default: 'results_summary.json').
        
        Returns
        -------
        str
            Path to summary file.
        """
        try:
            metrics_dir = os.path.join(self.results_dir, 'metrics')
            if not os.path.exists(metrics_dir):
                raise IOError(f"Metrics directory not found: {metrics_dir}")
            
            summary = {
                'dataset': dataset_name,
                'subjects': {},
                'aggregated_metrics': {}
            }
            
            # Load all metrics files for this dataset
            for filename in os.listdir(metrics_dir):
                if filename.startswith(dataset_name) and filename.endswith('_metrics.json'):
                    subject_id = filename.replace(f"{dataset_name}_", "").replace("_metrics.json", "")
                    filepath = os.path.join(metrics_dir, filename)
                    metrics = self.data_loader.load_metrics(filepath)
                    summary['subjects'][subject_id] = metrics
            
            # Compute aggregated metrics
            metric_keys = set()
            for metrics in summary['subjects'].values():
                metric_keys.update(metrics.keys())
            
            for metric_key in metric_keys:
                values = []
                for metrics in summary['subjects'].values():
                    if metric_key in metrics and isinstance(metrics[metric_key], (int, float)):
                        values.append(float(metrics[metric_key]))
                
                if values:
                    summary['aggregated_metrics'][metric_key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
            
            # Save summary
            output_path = os.path.join(self.results_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results summary created: {output_path}")
            return output_path
            
        except Exception as e:
            raise IOError(f"Failed to create results summary: {str(e)}")
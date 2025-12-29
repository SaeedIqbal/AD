#!/usr/bin/env python3
"""
Main preprocessing script for the four datasets in the AD meta-state dynamics study.

This script orchestrates the preprocessing pipeline for OpenNeuro ds004504, 
BioFIND, Mendeley Olfactory, and PREVENT-AD datasets, applying dataset-specific
preprocessing as defined in the methodology.
"""

import os
import sys
import yaml
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import preprocessing modules
from preprocessing.openneuro import OpenNeuroPreprocessor
from preprocessing.biofind import BioFINDPreprocessor
from preprocessing.mendeley import MendeleyOlfactoryPreprocessor
from preprocessing.prevent_ad import PREVENTADPreprocessor

# Import utility modules
from utils.io import DataSaver, DatasetConfig
from utils.source_reconstruction import SourceReconstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PreprocessingOrchestrator:
    """
    Orchestrates the complete preprocessing pipeline for all four datasets.
    
    This class handles file discovery, dataset-specific preprocessing,
    source reconstruction, and saving of intermediate results.
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = '/home/phd/results'
    ):
        """
        Initialize preprocessing orchestrator.
        
        Parameters
        ----------
        config_path : str
            Path to dataset configuration file.
        output_dir : str
            Base output directory for results.
        """
        self.config = DatasetConfig(config_path)
        self.data_saver = DataSaver(output_dir)
        self.output_dir = output_dir
        self.dataset_preprocessors = {
            'OpenNeuro_ds004504': self._preprocess_openneuro,
            'BioFIND': self._preprocess_biofind,
            'Mendeley_Olfactory': self._preprocess_mendeley,
            'PREVENT_AD': self._preprocess_prevent_ad
        }
    
    def _find_subject_files(self, dataset_name: str) -> List[str]:
        """
        Find all subject files for a given dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        
        Returns
        -------
        List[str]
            List of subject file paths.
        """
        dataset_path = self.config.get_dataset_path(dataset_name)
        dataset_info = self.config.get_dataset_info(dataset_name)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return []
        
        # Get supported file extensions for each dataset
        extensions = {
            'OpenNeuro_ds004504': ['.set', '.edf', '.mat'],
            'BioFIND': ['.fif'],
            'Mendeley_Olfactory': ['.mat', '.csv'],
            'PREVENT_AD': ['.vhdr', '.eeg', '.gdf', '.edf', '.bdf']
        }
        
        supported_ext = extensions.get(dataset_name, [])
        subject_files = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if any(file.endswith(ext) for ext in supported_ext):
                    subject_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(subject_files)} subject files for {dataset_name}")
        return sorted(subject_files)
    
    def _extract_subject_id(self, filepath: str, dataset_name: str) -> str:
        """
        Extract subject ID from filename.
        
        Parameters
        ----------
        filepath : str
            Path to subject file.
        dataset_name : str
            Name of the dataset.
        
        Returns
        -------
        str
            Subject ID.
        """
        filename = os.path.basename(filepath)
        
        if dataset_name == 'OpenNeuro_ds004504':
            # Extract from sub-XXXX format
            if 'sub-' in filename:
                return filename.split('sub-')[1].split('_')[0].split('.')[0]
        elif dataset_name == 'BioFIND':
            # Use filename without extension
            return filename.split('.')[0]
        elif dataset_name == 'Mendeley_Olfactory':
            # Use filename without extension
            return filename.split('.')[0]
        elif dataset_name == 'PREVENT_AD':
            # Extract from filename
            return filename.split('.')[0]
        
        # Fallback: use filename without extension
        return os.path.splitext(filename)[0]
    
    def _preprocess_openneuro(self, filepath: str) -> Optional[np.ndarray]:
        """Preprocess OpenNeuro ds004504 file."""
        try:
            preprocessor = OpenNeuroPreprocessor(filepath)
            return preprocessor.run()
        except Exception as e:
            logger.error(f"OpenNeuro preprocessing failed for {filepath}: {str(e)}")
            return None
    
    def _preprocess_biofind(self, filepath: str) -> Optional[np.ndarray]:
        """Preprocess BioFIND file."""
        try:
            preprocessor = BioFINDPreprocessor(filepath)
            return preprocessor.run()
        except Exception as e:
            logger.error(f"BioFIND preprocessing failed for {filepath}: {str(e)}")
            return None
    
    def _preprocess_mendeley(self, filepath: str) -> Optional[np.ndarray]:
        """Preprocess Mendeley Olfactory file."""
        try:
            preprocessor = MendeleyOlfactoryPreprocessor(filepath)
            return preprocessor.run()
        except Exception as e:
            logger.error(f"Mendeley preprocessing failed for {filepath}: {str(e)}")
            return None
    
    def _preprocess_prevent_ad(self, filepath: str) -> Optional[np.ndarray]:
        """Preprocess PREVENT-AD file."""
        try:
            preprocessor = PREVENTADPreprocessor(filepath)
            return preprocessor.run()
        except Exception as e:
            logger.error(f"PREVENT-AD preprocessing failed for {filepath}: {str(e)}")
            return None
    
    def _reconstruct_sources(
        self,
        sensor_ np.ndarray,
        dataset_name: str,
        subject_id: str
    ) -> Optional[np.ndarray]:
        """
        Reconstruct source time series from sensor data.
        
        Parameters
        ----------
        sensor_data : np.ndarray
            Preprocessed sensor data.
        dataset_name : str
            Name of the dataset.
        subject_id : str
            Subject ID.
        
        Returns
        -------
        np.ndarray or None
            Source time series or None if failed.
        """
        try:
            n_channels = sensor_data.shape[0]
            
            # Determine system type for PREVENT-AD
            system_type = 'auto'
            if dataset_name == 'PREVENT_AD':
                if n_channels == 32:
                    system_type = 'gtec'
                elif n_channels == 64:
                    system_type = 'brainamp'
            
            reconstructor = SourceReconstructor(dataset_name)
            source_data = reconstructor.reconstruct_sources(
                sensor_data, n_channels, system_type
            )
            
            logger.info(f"Source reconstruction completed for {dataset_name} {subject_id}")
            return source_data
            
        except Exception as e:
            logger.error(f"Source reconstruction failed for {dataset_name} {subject_id}: {str(e)}")
            return None
    
    def process_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Process all subjects in a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process.
        
        Returns
        -------
        dict
            Processing results summary.
        """
        logger.info(f"Starting preprocessing for dataset: {dataset_name}")
        
        subject_files = self._find_subject_files(dataset_name)
        if not subject_files:
            logger.warning(f"No subject files found for {dataset_name}")
            return {'dataset': dataset_name, 'processed': 0, 'failed': 0, 'errors': []}
        
        processed_count = 0
        failed_count = 0
        errors = []
        
        for filepath in subject_files:
            try:
                subject_id = self._extract_subject_id(filepath, dataset_name)
                logger.info(f"Processing {dataset_name} subject: {subject_id}")
                
                # Preprocess sensor data
                preprocessor_func = self.dataset_preprocessors[dataset_name]
                sensor_data = preprocessor_func(filepath)
                
                if sensor_data is None:
                    failed_count += 1
                    errors.append(f"{subject_id}: Preprocessing failed")
                    continue
                
                # Save preprocessed sensor data
                dataset_info = self.config.get_dataset_info(dataset_name)
                self.data_saver.save_preprocessed_data(
                    dataset_name,
                    subject_id,
                    sensor_data,
                    dataset_info.get('sfreq', 256.0),
                    dataset_info.get('ch_names', [f'ch{i}' for i in range(sensor_data.shape[0])])
                )
                
                # Reconstruct sources
                source_data = self._reconstruct_sources(sensor_data, dataset_name, subject_id)
                if source_data is None:
                    failed_count += 1
                    errors.append(f"{subject_id}: Source reconstruction failed")
                    continue
                
                # Save source data (as preprocessed data with 68 channels)
                self.data_saver.save_preprocessed_data(
                    dataset_name,
                    f"{subject_id}_source",
                    source_data,
                    dataset_info.get('sfreq', 256.0),
                    [f'ROI_{i}' for i in range(68)]
                )
                
                processed_count += 1
                logger.info(f"Completed processing {dataset_name} {subject_id}")
                
            except Exception as e:
                failed_count += 1
                error_msg = f"{subject_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Processing failed for {dataset_name} {subject_id}: {str(e)}")
        
        results = {
            'dataset': dataset_name,
            'processed': processed_count,
            'failed': failed_count,
            'total': len(subject_files),
            'errors': errors
        }
        
        logger.info(f"Completed preprocessing for {dataset_name}: {processed_count}/{len(subject_files)} subjects")
        return results
    
    def run_all_datasets(self) -> Dict[str, Any]:
        """
        Run preprocessing on all configured datasets.
        
        Returns
        -------
        dict
            Comprehensive results summary.
        """
        logger.info("Starting preprocessing pipeline for all datasets")
        
        datasets = self.config.list_datasets()
        all_results = {}
        overall_processed = 0
        overall_failed = 0
        
        for dataset_name in datasets:
            try:
                results = self.process_dataset(dataset_name)
                all_results[dataset_name] = results
                overall_processed += results['processed']
                overall_failed += results['failed']
            except Exception as e:
                logger.error(f"Dataset {dataset_name} failed completely: {str(e)}")
                all_results[dataset_name] = {
                    'dataset': dataset_name,
                    'processed': 0,
                    'failed': 0,
                    'total': 0,
                    'errors': [f'Complete failure: {str(e)}']
                }
        
        # Save overall results
        summary = {
            'overall': {
                'processed': overall_processed,
                'failed': overall_failed,
                'datasets': all_results
            }
        }
        
        summary_path = os.path.join(self.output_dir, 'preprocessing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Preprocessing pipeline completed. Results saved to {summary_path}")
        return summary


def main():
    """Main entry point for preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess AD meta-state datasets')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/dataset_paths.yaml',
        help='Path to dataset configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/phd/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        help='Specific dataset to process (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = PreprocessingOrchestrator(args.config, args.output_dir)
        
        if args.dataset == 'all':
            results = orchestrator.run_all_datasets()
        else:
            results = orchestrator.process_dataset(args.dataset)
        
        print("\nPreprocessing Summary:")
        print("=" * 50)
        if isinstance(results, dict) and 'overall' in results:
            for dataset, result in results['overall']['datasets'].items():
                print(f"{dataset}: {result['processed']}/{result['total']} processed")
        else:
            print(f"Processed: {results.get('processed', 0)}/{results.get('total', 0)}")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
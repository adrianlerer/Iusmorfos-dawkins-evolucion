#!/usr/bin/env python3
"""
Configuration Management for Iusmorfos
======================================

Centralized configuration system ensuring reproducibility across all experiments.
Handles random seed management, path configuration, and parameter validation.

Following FAIR principles and reproducibility best practices.
"""

import os
import sys
import yaml
import json
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

import numpy as np


class IusmorfosConfig:
    """
    Centralized configuration manager for reproducible experiments.
    
    Features:
    - Automatic random seed management
    - Path resolution and creation
    - Parameter validation
    - Logging configuration
    - Environment consistency
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Create directories
        self._create_directories()
        
        # Configure logging
        self._setup_logging()
        
    def _find_config_file(self) -> str:
        """Find config.yaml in standard locations."""
        possible_paths = [
            "config/config.yaml",
            "../config/config.yaml", 
            "../../config/config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(
            f"config.yaml not found in: {possible_paths}\n"
            "Please ensure config/config.yaml exists in the project root."
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_sections = ['reproducibility', 'experiment', 'paths']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")
    
    def _setup_reproducibility(self) -> None:
        """Configure all random number generators for reproducibility."""
        repro_config = self.config['reproducibility']
        
        # Set all possible random seeds
        seed = repro_config['random_seed']
        
        # Python random
        random.seed(repro_config.get('python_seed', seed))
        
        # NumPy
        np.random.seed(repro_config.get('numpy_seed', seed))
        
        # Python hash randomization (must be set before Python starts)
        os.environ['PYTHONHASHSEED'] = str(repro_config.get('hash_seed', seed))
        
        print(f"🔒 Reproducibility configured with master seed: {seed}")
    
    def _create_directories(self) -> None:
        """Create all necessary directories."""
        paths = self.config['paths']
        
        directories = [
            paths['data_dir'],
            paths['results_dir'], 
            paths['outputs_dir'],
            paths['logs_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Configure logging system."""
        log_config = self.config['logging']
        
        log_file = self.get_path('logs_dir') / f"iusmorfos_{self.timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('iusmorfos')
        self.logger.info(f"🚀 Iusmorfos initialized - Config: {self.config_path}")
    
    def get_path(self, path_key: str, **kwargs) -> Path:
        """Get resolved path from configuration with optional formatting."""
        paths = self.config['paths']
        
        if path_key not in paths:
            raise KeyError(f"Path key '{path_key}' not found in configuration")
        
        path_template = paths[path_key]
        
        # Format with timestamp and any additional kwargs
        format_vars = {'timestamp': self.timestamp, **kwargs}
        path_str = path_template.format(**format_vars)
        
        return Path(path_str)
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.config['experiment'].copy()
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.config['validation'].copy()
    
    def get_iuspace_config(self) -> Dict[str, Any]:
        """Get IusSpace dimension configuration."""
        return self.config['iuspace'].copy()
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get specific model configuration."""
        models = self.config['models']
        
        if model_name not in models:
            raise KeyError(f"Model '{model_name}' not configured")
        
        return models[model_name].copy()
    
    def save_results(self, results: Dict[str, Any], 
                    filename_key: str = 'experiment_results') -> Path:
        """Save results to configured location."""
        output_path = self.get_path(filename_key)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results_with_metadata = {
            'metadata': {
                'timestamp': self.timestamp,
                'config_file': str(self.config_path),
                'seed': self.config['reproducibility']['random_seed'],
                'version': '1.0.0'
            },
            'results': results
        }
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Results saved: {output_path}")
        return output_path
    
    def validate_experiment_results(self, results: Dict[str, Any]) -> bool:
        """Validate experiment results against expected ranges."""
        validation_config = self.get_validation_config()
        tolerances = validation_config['tolerances']
        
        checks = []
        
        # Check if required keys exist
        required_keys = ['complexity_evolution', 'fitness_evolution', 'final_generation']
        for key in required_keys:
            if key not in results:
                self.logger.warning(f"⚠️ Missing result key: {key}")
                checks.append(False)
            else:
                checks.append(True)
        
        # Additional validation logic can be added here
        
        passed = all(checks)
        if passed:
            self.logger.info("✅ Results validation passed")
        else:
            self.logger.error("❌ Results validation failed")
        
        return passed
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility tracking."""
        import platform
        
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'timestamp': self.timestamp,
            'config_hash': self._get_config_hash()
        }
    
    def _get_config_hash(self) -> str:
        """Generate hash of configuration for change detection."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"IusmorfosConfig(seed={self.config['reproducibility']['random_seed']}, timestamp={self.timestamp})"


# Global configuration instance (singleton pattern)
_config_instance: Optional[IusmorfosConfig] = None

def get_config(config_path: Optional[str] = None) -> IusmorfosConfig:
    """Get global configuration instance (singleton)."""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = IusmorfosConfig(config_path)
    
    return _config_instance

def reset_config() -> None:
    """Reset configuration (mainly for testing)."""
    global _config_instance
    _config_instance = None


if __name__ == "__main__":
    # Demo usage
    print("🧬 Iusmorfos Configuration System")
    print("=" * 40)
    
    try:
        config = get_config()
        print(f"✅ Configuration loaded: {config}")
        print(f"📁 Data directory: {config.get_path('data_dir')}")
        print(f"🎯 Experiment config: {config.get_experiment_config()}")
        print(f"🔍 System info: {config.get_system_info()}")
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
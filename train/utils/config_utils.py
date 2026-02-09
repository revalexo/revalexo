# utils/config_utils.py

import os
import yaml
import torch
import random
import numpy as np
from typing import Dict, Any, Optional, Union


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            config['__config_file__'] = os.path.abspath(config_path)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")


def parse_nested_config(config: Dict[str, Any], key_path: str, default=None) -> Any:
    """
    Safely access nested configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Nested key path using dot notation (e.g., 'training.optimizer.params.lr')
        default: Default value to return if the key path is not found
        
    Returns:
        Value at the specified key path or default if not found
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def setup_seed(seed: Optional[int] = None) -> int:
    """
    Set up random seed for reproducibility.
    
    Args:
        seed: Random seed to use, if None a random seed will be generated
        
    Returns:
        The seed that was set
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return seed


def resolve_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in the configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for resolving relative paths, if None use current directory
        
    Returns:
        Configuration with resolved absolute paths
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    def _resolve_paths_recursive(cfg, current_path=""):
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if isinstance(value, (dict, list)):
                    cfg[key] = _resolve_paths_recursive(value, current_path + "." + key if current_path else key)
                elif isinstance(value, str) and key.endswith(('_path', '_dir', '_file')) and not os.path.isabs(value):
                    cfg[key] = os.path.normpath(os.path.join(base_dir, value))
        elif isinstance(cfg, list):
            for i, item in enumerate(cfg):
                if isinstance(item, (dict, list)):
                    cfg[i] = _resolve_paths_recursive(item, current_path + f"[{i}]")
        return cfg
    
    return _resolve_paths_recursive(config.copy())


def get_device(no_cuda: bool = False, rank: int = 0) -> torch.device:
    """
    Get PyTorch device to use.
    
    Args:
        no_cuda: If True, force CPU usage even if CUDA is available
        rank: Process rank in distributed training to select specific GPU
        
    Returns:
        PyTorch device to use
    """
    if no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        # In distributed setting, assign specific GPU based on rank
        device = torch.device(f"cuda:{rank}")
    
    return device


def handle_scientific_notation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert scientific notation strings to float values in config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with scientific notation strings converted to floats
    """
    def _convert_recursive(cfg):
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if isinstance(value, (dict, list)):
                    cfg[key] = _convert_recursive(value)
                elif isinstance(value, str):
                    try:
                        if 'e' in value.lower():
                            cfg[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        elif isinstance(cfg, list):
            for i, item in enumerate(cfg):
                if isinstance(item, (dict, list)):
                    cfg[i] = _convert_recursive(item)
                elif isinstance(item, str):
                    try:
                        if 'e' in item.lower():
                            cfg[i] = float(item)
                    except (ValueError, TypeError):
                        pass
        return cfg
    
    return _convert_recursive(config.copy())
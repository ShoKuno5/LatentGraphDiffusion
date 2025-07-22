"""
WandB configuration loader for secure credential management.
"""
import os
import configparser
import logging
from pathlib import Path


def load_wandb_config(profile='default', config_file='.wandbrc'):
    """
    Load WandB configuration from .wandbrc file or environment variables.
    
    Args:
        profile (str): Profile section to load from .wandbrc
        config_file (str): Path to config file relative to project root
    
    Returns:
        dict: WandB configuration with keys: entity, project, api_key
    """
    # Find project root (directory containing the config file)
    current_dir = Path(__file__).parent
    project_root = current_dir
    
    # Walk up directories to find .wandbrc
    while project_root.parent != project_root:
        if (project_root / config_file).exists():
            break
        project_root = project_root.parent
    
    config_path = project_root / config_file
    config = {}
    
    # Try to load from config file first
    if config_path.exists():
        try:
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            if profile in parser:
                section = parser[profile]
                config['entity'] = section.get('entity', '')
                config['project'] = section.get('project', 'LatentGraphDiffusion-ZINC')
                config['api_key'] = section.get('api_key', '')
                logging.info(f"Loaded WandB config from {config_path} (profile: {profile})")
            else:
                logging.warning(f"Profile '{profile}' not found in {config_path}")
        except Exception as e:
            logging.error(f"Error reading {config_path}: {e}")
    
    # Fallback to environment variables
    if not config.get('api_key'):
        config['api_key'] = os.getenv('WANDB_API_KEY', '')
    if not config.get('entity'):
        config['entity'] = os.getenv('WANDB_ENTITY', '')
    if not config.get('project'):
        config['project'] = os.getenv('WANDB_PROJECT', 'LatentGraphDiffusion-ZINC')
    
    # Validate required fields
    if not config.get('api_key'):
        logging.warning("No WandB API key found. Set it in .wandbrc or WANDB_API_KEY env var")
    
    return config


def setup_wandb_env(profile='default'):
    """
    Setup WandB environment variables from config file.
    This ensures wandb.init() can find the credentials.
    
    Args:
        profile (str): Profile to load from .wandbrc
    """
    config = load_wandb_config(profile)
    
    if config.get('api_key'):
        os.environ['WANDB_API_KEY'] = config['api_key']
    if config.get('entity'):
        os.environ['WANDB_ENTITY'] = config['entity']
    if config.get('project'):
        os.environ['WANDB_PROJECT'] = config['project']
    
    return config
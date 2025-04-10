"""
Configuration utility module for environment setup and management.
"""

import os
import logging
from dotenv import load_dotenv
from .logger import get_logger

def setup_environment() -> bool:
    """
    Load environment variables and setup logging.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Load environment variables from .env file if present
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        load_dotenv(dotenv_path)
        
        # Setup logging
        try:
            logger = get_logger('main')
        except Exception as e:
            logging.error(f"Logging setup failed: {str(e)}")
            return False
            
        logger.info("Environment setup complete.")
        
        # Create necessary directories
        for dir_name in ['data', 'output', 'logs']:
            dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Environment setup failed: {str(e)}")
        return False 
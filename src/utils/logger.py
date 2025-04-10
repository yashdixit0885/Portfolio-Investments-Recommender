"""
Logging utility module for consistent logging across the application.
"""

import logging
import os
from datetime import datetime

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name (str, optional): Name of the logger. If None, returns the root logger.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logger instance
    logger = logging.getLogger(name) if name else logging.getLogger()
    
    # Only configure if handlers haven't been set up
    if not logger.handlers:
        # Set logging level
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 
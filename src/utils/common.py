import os
import json
import logging
import datetime
import pytz
import pandas as pd
import shutil
import re
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
from pathlib import Path

def setup_logging(component_name: str) -> logging.Logger:
    """Set up logging for a component."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(component_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(f'logs/{component_name}.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_current_time(timezone=None):
    """Get current time in specified timezone or UTC."""
    if timezone == '':
        raise pytz.exceptions.UnknownTimeZoneError('Empty timezone string')
        
    if not timezone:
        return datetime.now(pytz.UTC)
    
    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz)
    except pytz.exceptions.UnknownTimeZoneError:
        raise

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_to_json(data: Union[Dict[str, Any], List[Any]], filepath: str) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Dictionary or list to save
        filepath: Full path to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=DateTimeEncoder)
        return True
    except Exception as e:
        logging.error(f"Error saving to JSON: {str(e)}")
        return False

def load_from_json(filepath: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Load data from JSON file
    
    Args:
        filepath: Full path to the file
        
    Returns:
        Optional[Union[Dict[str, Any], List[Any]]]: Loaded data or None if error
    """
    try:
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading from JSON: {str(e)}")
        return None

def format_currency(amount: float) -> str:
    """Format currency amount"""
    if amount is None:
        return '$0.00'
    if isinstance(amount, (int, float)):
        if amount != amount:  # Check for NaN
            return '$nan'
        if amount == float('inf'):
            return '$inf'
        if amount == float('-inf'):
            return '$-inf'
        return f"${amount:,.2f}"
    return '$0.00'

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / abs(old_value)) * 100

def validate_email(email):
    """Validate email format."""
    if not email:
        return False
    
    try:
        # Split email into local and domain parts
        local, domain = email.split('@')
        
        # Check local part
        if not local or len(local) > 64:
            return False
        
        # Check for consecutive dots
        if '..' in local or '..' in domain:
            return False
        
        # Check domain part
        if not domain or len(domain) > 255:
            return False
        
        # Check domain parts
        domain_parts = domain.split('.')
        if len(domain_parts) < 2:
            return False
            
        # Check TLD length (last part)
        tld = domain_parts[-1]
        if not tld or len(tld) < 2:
            return False
            
        # Allow international domains and regular domains
        for part in domain_parts:
            if not part:
                return False
            # Check if it's a punycode domain
            try:
                part.encode('ascii')
            except UnicodeEncodeError:
                # International domain part, which is valid
                continue
            # Regular ASCII domain part validation
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$', part):
                return False
                
        return True
    except Exception:
        return False

def save_recommendations_to_csv(recommendations: List[Dict[str, Any]], output_dir: str = "Output") -> str:
    """
    Save trade recommendations to a CSV file.
    
    Args:
        recommendations: List of recommendation dictionaries
        output_dir: Directory to save the CSV file
        
    Returns:
        str: Path to the saved CSV file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if not recommendations:
            # Create empty file
            filepath = os.path.join(output_dir, 'recommendations.csv')
            with open(filepath, 'w') as f:
                f.write('ticker,name,action,score,position_size,price,market_cap,price_change,volume_change,rationale\n')
            return filepath
        
        # Convert to DataFrame
        df = pd.DataFrame(recommendations)
        
        # Save to CSV
        filepath = os.path.join(output_dir, 'recommendations.csv')
        df.to_csv(filepath, index=False)
        return filepath
    except Exception as e:
        logging.error(f"Error saving recommendations: {str(e)}")
        raise

def archive_and_cleanup_files(output_dir, archive_dir):
    """Archive files at 5 PM MT and cleanup at 7 AM MT."""
    try:
        current_time = get_current_time('America/Denver')
        hour = current_time.hour
        
        if hour == 17:  # 5 PM MT
            # Archive files
            date_str = current_time.strftime('%Y%m%d')
            date_dir = os.path.join(archive_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            for filename in os.listdir(output_dir):
                if filename.endswith(('.csv', '.json')):
                    src = os.path.join(output_dir, filename)
                    dst = os.path.join(date_dir, filename)
                    try:
                        if os.access(src, os.W_OK):
                            os.rename(src, dst)
                        else:
                            logging.warning(f"Could not archive read-only file: {src}")
                    except (PermissionError, OSError) as e:
                        logging.warning(f"Could not archive file {src}: {str(e)}")
                    
        elif hour == 7:  # 7 AM MT
            # Cleanup files
            for filename in os.listdir(output_dir):
                if filename.endswith(('.csv', '.json')):
                    filepath = os.path.join(output_dir, filename)
                    try:
                        if os.access(filepath, os.W_OK):
                            os.remove(filepath)
                        else:
                            logging.warning(f"Could not remove read-only file: {filepath}")
                    except (PermissionError, OSError) as e:
                        logging.warning(f"Could not remove file {filepath}: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error in archive_and_cleanup_files: {str(e)}")
        raise
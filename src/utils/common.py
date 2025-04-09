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

def setup_logging(component_name: str) -> logging.Logger:
    """Setup logging for a component"""
    logger = logging.getLogger(component_name)
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'logs/{component_name}.log')
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def get_current_time(timezone: Optional[str] = None) -> datetime:
    """
    Get the current time in the specified timezone or UTC if no timezone is provided.
    
    Args:
        timezone: Optional timezone string (e.g., 'America/New_York', 'America/Denver')
                 If None, returns UTC time
    
    Returns:
        datetime: Current time in specified timezone or UTC
    """
    if timezone:
        return datetime.now(pytz.timezone(timezone))
    return datetime.now(pytz.UTC)

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
        
        # Save data using custom encoder
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, cls=DateTimeEncoder)
            
        return True
        
    except Exception as e:
        logger = logging.getLogger('utils')
        logger.error(f"Error saving to {filepath}: {str(e)}")
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
            logger = logging.getLogger('utils')
            logger.warning(f"File not found: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return data
        
    except json.JSONDecodeError as e:
        logger = logging.getLogger('utils')
        logger.error(f"Error decoding JSON from {filepath}: {str(e)}")
        return None
        
    except Exception as e:
        logger = logging.getLogger('utils')
        logger.error(f"Error loading from {filepath}: {str(e)}")
        return None

def format_currency(amount: float) -> str:
    """Format currency amount"""
    return f"${amount:,.2f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Handle archiving and cleanup based on time
        archive_and_cleanup_files(output_dir)
            
        # Format the data for CSV
        csv_data = []
        for rec in recommendations:
            csv_data.append({
                'Ticker': rec.get('ticker', ''),
                'Company': rec.get('name', ''),
                'Action': rec.get('action', ''),
                'Score': rec.get('score', 0.0),
                'Position Size': rec.get('position_size', 0.0),
                'Current Price': rec.get('price', 0.0),
                'Market Cap': rec.get('market_cap', 0),
                'Price Change': rec.get('price_change', 0.0),
                'Volume Change': rec.get('volume_change', 0.0),
                'Rationale': rec.get('rationale', '')
            })
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        current_date = get_current_time().strftime('%Y%m%d')
        filename = f'trade_recommendations_{current_date}.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Saved {len(recommendations)} recommendations to {filepath}")
        
        return filepath
        
    except Exception as e:
        logging.error(f"Error saving recommendations to CSV: {str(e)}")
        raise

def archive_and_cleanup_files(output_dir: str = "Output", archive_dir: str = "Archive") -> None:
    """
    Archive files at 5 PM MT and clean up at 7 AM MT.
    
    Args:
        output_dir: Directory containing output files
        archive_dir: Directory for archived files
    """
    try:
        current_time = get_current_time('America/Denver')  # Mountain Time
        
        # Create archive directory if it doesn't exist
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
            
        # At 5 PM MT - Archive files
        if current_time.hour == 17:
            for file in os.listdir(output_dir):
                if file.endswith('.csv') or file.endswith('.json'):
                    # Create archive subfolder with date
                    date_folder = os.path.join(archive_dir, current_time.strftime('%Y%m%d'))
                    if not os.path.exists(date_folder):
                        os.makedirs(date_folder)
                    
                    # Move file to archive
                    src = os.path.join(output_dir, file)
                    dst = os.path.join(date_folder, file)
                    shutil.move(src, dst)
                    logging.info(f"Archived {file} to {date_folder}")
                    
        # At 7 AM MT - Clean up output directory
        elif current_time.hour == 7:
            for file in os.listdir(output_dir):
                if file.endswith('.csv') or file.endswith('.json'):
                    os.remove(os.path.join(output_dir, file))
                    logging.info(f"Cleaned up {file} from {output_dir}")
                    
    except Exception as e:
        logging.error(f"Error in archive_and_cleanup_files: {str(e)}")
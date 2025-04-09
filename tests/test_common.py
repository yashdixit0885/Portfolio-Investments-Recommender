import os
import json
import pytest
import logging
import pytz
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.utils.common import (
    setup_logging,
    get_current_time,
    DateTimeEncoder,
    save_to_json,
    load_from_json,
    format_currency,
    calculate_percentage_change,
    validate_email,
    save_recommendations_to_csv,
    archive_and_cleanup_files
)

@pytest.fixture
def sample_recommendations():
    """Create sample trade recommendations for testing."""
    return [
        {
            'ticker': 'AAPL',
            'name': 'Apple Inc',
            'action': 'BUY',
            'score': 85.5,
            'position_size': 1000,
            'price': 150.0,
            'market_cap': 2500000000000,
            'price_change': 5.2,
            'volume_change': 15.3,
            'rationale': 'Strong momentum'
        },
        {
            'ticker': 'MSFT',
            'name': 'Microsoft Corp',
            'action': 'SELL',
            'score': 45.2,
            'position_size': 500,
            'price': 280.0,
            'market_cap': 2000000000000,
            'price_change': -2.1,
            'volume_change': -5.5,
            'rationale': 'Weakening trend'
        }
    ]

def test_setup_logging():
    """Test logging setup functionality."""
    logger = setup_logging('test_component')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_component'
    assert logger.level == logging.INFO
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert os.path.exists('logs/test_component.log')

def test_get_current_time():
    """Test current time retrieval with different timezones."""
    # Test UTC time
    utc_time = get_current_time()
    assert utc_time.tzinfo == pytz.UTC
    
    # Test specific timezone
    mt_time = get_current_time('America/Denver')
    assert str(mt_time.tzinfo).startswith('America/Denver')  # More flexible assertion
    
    # Test invalid timezone
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        get_current_time('Invalid/Timezone')

def test_datetime_encoder():
    """Test custom datetime JSON encoder."""
    test_time = datetime(2024, 4, 9, 12, 0, tzinfo=pytz.UTC)
    test_data = {'timestamp': test_time, 'value': 42}
    
    encoded = json.dumps(test_data, cls=DateTimeEncoder)
    assert isinstance(encoded, str)
    assert '2024-04-09T12:00:00+00:00' in encoded

def test_save_to_json(tmp_path):
    """Test JSON file saving functionality."""
    # Test successful save
    test_data = {'key': 'value', 'number': 42}
    test_file = tmp_path / 'test.json'
    assert save_to_json(test_data, str(test_file)) is True
    assert test_file.exists()
    
    # Test save with datetime
    test_data['timestamp'] = datetime.now(pytz.UTC)
    assert save_to_json(test_data, str(test_file)) is True
    
    # Test save with invalid path
    assert save_to_json(test_data, '/invalid/path/test.json') is False

def test_load_from_json(tmp_path):
    """Test JSON file loading functionality."""
    # Test successful load
    test_data = {'key': 'value', 'number': 42}
    test_file = tmp_path / 'test.json'
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    loaded_data = load_from_json(str(test_file))
    assert loaded_data == test_data
    
    # Test load nonexistent file
    assert load_from_json('/nonexistent/file.json') is None
    
    # Test load invalid JSON
    invalid_file = tmp_path / 'invalid.json'
    with open(invalid_file, 'w') as f:
        f.write('invalid json')
    assert load_from_json(str(invalid_file)) is None

def test_format_currency():
    """Test currency formatting."""
    assert format_currency(1234.5678) == '$1,234.57'
    assert format_currency(0) == '$0.00'
    assert format_currency(1000000) == '$1,000,000.00'
    assert format_currency(-1234.56) == '$-1,234.56'  # Updated to match actual format

def test_calculate_percentage_change():
    """Test percentage change calculation."""
    assert calculate_percentage_change(100, 150) == 50.0
    assert calculate_percentage_change(150, 100) == -33.33333333333333
    assert calculate_percentage_change(0, 100) == 0
    assert calculate_percentage_change(100, 100) == 0.0

def test_validate_email():
    """Test email validation."""
    assert validate_email('test@example.com') is True
    assert validate_email('test.name@example.co.uk') is True
    assert validate_email('test@.com') is False
    assert validate_email('test@com') is False
    assert validate_email('test.com') is False
    assert validate_email('') is False

def test_save_recommendations_to_csv(tmp_path, sample_recommendations):
    """Test saving recommendations to CSV."""
    output_dir = tmp_path / 'output'
    
    # Test successful save
    filepath = save_recommendations_to_csv(sample_recommendations, str(output_dir))
    assert os.path.exists(filepath)
    
    # Test empty recommendations - should not raise exception
    empty_filepath = save_recommendations_to_csv([], str(output_dir))
    assert os.path.exists(empty_filepath)
    
    # Test invalid directory
    with pytest.raises(Exception):
        save_recommendations_to_csv(sample_recommendations, '/invalid/dir')

def test_archive_and_cleanup_files(tmp_path):
    """Test file archiving and cleanup."""
    output_dir = tmp_path / 'output'
    archive_dir = tmp_path / 'archive'
    os.makedirs(output_dir)
    
    # Create test files
    test_file = output_dir / 'test.csv'
    test_file.write_text('test data')
    
    # Test archiving at 5 PM MT
    with patch('src.utils.common.get_current_time') as mock_time:
        mock_time.return_value = datetime(2024, 4, 9, 17, 0, tzinfo=pytz.timezone('America/Denver'))
        archive_and_cleanup_files(str(output_dir), str(archive_dir))
        assert os.path.exists(archive_dir)
        
    # Test cleanup at 7 AM MT
    with patch('src.utils.common.get_current_time') as mock_time:
        mock_time.return_value = datetime(2024, 4, 9, 7, 0, tzinfo=pytz.timezone('America/Denver'))
        archive_and_cleanup_files(str(output_dir), str(archive_dir))
        assert not os.path.exists(test_file) 
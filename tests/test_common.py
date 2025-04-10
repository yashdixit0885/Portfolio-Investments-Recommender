import os
import json
import pytest
import logging
import pytz
import pandas as pd
import numpy as np
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
    # Test basic setup
    logger = setup_logging('test_component')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_component'
    assert logger.level == logging.INFO
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert os.path.exists('logs/test_component.log')
    
    # Test with existing log file
    logger2 = setup_logging('test_component')
    assert logger2.name == 'test_component'
    assert os.path.exists('logs/test_component.log')
    
    # Test with different log level
    logger3 = setup_logging('test_component_debug')
    logger3.setLevel(logging.DEBUG)
    assert logger3.level == logging.DEBUG

def test_get_current_time():
    """Test current time retrieval with different timezones."""
    # Test UTC time
    utc_time = get_current_time()
    assert utc_time.tzinfo == pytz.UTC
    
    # Test specific timezones
    timezones = ['America/New_York', 'Europe/London', 'Asia/Tokyo', 'Australia/Sydney']
    for tz in timezones:
        local_time = get_current_time(tz)
        assert str(local_time.tzinfo).startswith(tz)
    
    # Test empty timezone string
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        get_current_time('')
    
    # Test invalid timezone
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        get_current_time('Invalid/Timezone')

def test_datetime_encoder():
    """Test custom datetime JSON encoder."""
    # Test basic datetime
    test_time = datetime(2024, 4, 9, 12, 0, tzinfo=pytz.UTC)
    test_data = {'timestamp': test_time, 'value': 42}
    encoded = json.dumps(test_data, cls=DateTimeEncoder)
    assert isinstance(encoded, str)
    assert '2024-04-09T12:00:00+00:00' in encoded
    
    # Test nested datetime objects
    nested_data = {
        'outer': {
            'timestamp': test_time,
            'inner': {
                'time': test_time
            }
        }
    }
    encoded = json.dumps(nested_data, cls=DateTimeEncoder)
    assert '2024-04-09T12:00:00+00:00' in encoded
    
    # Test with other types
    mixed_data = {
        'timestamp': test_time,
        'string': 'test',
        'number': 42,
        'list': [1, 2, 3],
        'dict': {'key': 'value'}
    }
    encoded = json.dumps(mixed_data, cls=DateTimeEncoder)
    assert isinstance(encoded, str)

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
    
    # Test save with nested datetime
    nested_data = {
        'outer': {
            'timestamp': datetime.now(pytz.UTC),
            'inner': {
                'time': datetime.now(pytz.UTC)
            }
        }
    }
    assert save_to_json(nested_data, str(test_file)) is True
    
    # Test save with large data
    large_data = {'key' + str(i): 'value' + str(i) for i in range(1000)}
    assert save_to_json(large_data, str(test_file)) is True
    
    # Test save with invalid data
    invalid_data = {'key': object()}  # Non-serializable object
    with pytest.raises(TypeError):
        save_to_json(invalid_data, str(test_file))
    
    # Test save with invalid path
    with pytest.raises(OSError):
        save_to_json(test_data, '/invalid/path/test.json')

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
    
    # Test load empty file
    empty_file = tmp_path / 'empty.json'
    with open(empty_file, 'w') as f:
        f.write('')
    assert load_from_json(str(empty_file)) is None
    
    # Test load corrupted file
    corrupted_file = tmp_path / 'corrupted.json'
    with open(corrupted_file, 'w') as f:
        f.write('{"key": "value"')  # Missing closing brace
    assert load_from_json(str(corrupted_file)) is None

def test_format_currency():
    """Test currency formatting."""
    # Test basic formatting
    assert format_currency(1234.5678) == '$1,234.57'
    assert format_currency(0) == '$0.00'
    assert format_currency(1000000) == '$1,000,000.00'
    assert format_currency(-1234.56) == '$-1,234.56'
    
    # Test very large numbers
    assert format_currency(1234567890123.45) == '$1,234,567,890,123.45'
    
    # Test very small numbers
    assert format_currency(0.0001) == '$0.00'
    assert format_currency(0.001) == '$0.00'
    assert format_currency(0.01) == '$0.01'
    
    # Test with NaN and infinity
    assert format_currency(float('nan')) == '$nan'
    assert format_currency(float('inf')) == '$inf'
    assert format_currency(float('-inf')) == '$-inf'

def test_calculate_percentage_change():
    """Test percentage change calculation."""
    # Test basic calculations
    assert calculate_percentage_change(100, 150) == 50.0
    assert calculate_percentage_change(150, 100) == -33.33333333333333
    assert calculate_percentage_change(0, 100) == 0
    assert calculate_percentage_change(100, 100) == 0.0
    
    # Test with negative values
    assert calculate_percentage_change(-100, -150) == -50.0
    assert calculate_percentage_change(-150, -100) == 33.33333333333333
    
    # Test with zero new value
    assert calculate_percentage_change(100, 0) == -100.0
    
    # Test with very small values
    assert calculate_percentage_change(0.0001, 0.0002) == 100.0
    assert calculate_percentage_change(0.0002, 0.0001) == -50.0

def test_validate_email():
    """Test email validation."""
    # Test valid emails
    assert validate_email('test@example.com') is True
    assert validate_email('test.name@example.co.uk') is True
    assert validate_email('test+label@example.com') is True
    assert validate_email('test@subdomain.example.com') is True
    assert validate_email('test@example.io') is True
    
    # Test international domains
    assert validate_email('test@example.中国') is True
    assert validate_email('test@example.рф') is True
    
    # Test invalid emails
    assert validate_email('test@.com') is False
    assert validate_email('test@com') is False
    assert validate_email('test.com') is False
    assert validate_email('') is False
    assert validate_email('test@') is False
    assert validate_email('@example.com') is False
    assert validate_email('test@example..com') is False
    assert validate_email('test@example.c') is False

def test_save_recommendations_to_csv(tmp_path, sample_recommendations):
    """Test saving recommendations to CSV."""
    output_dir = tmp_path / 'output'
    
    # Test successful save
    filepath = save_recommendations_to_csv(sample_recommendations, str(output_dir))
    assert os.path.exists(filepath)
    
    # Test empty recommendations
    empty_filepath = save_recommendations_to_csv([], str(output_dir))
    assert os.path.exists(empty_filepath)
    
    # Test with missing fields
    incomplete_recommendations = [{'ticker': 'AAPL'}]  # Missing other fields
    filepath = save_recommendations_to_csv(incomplete_recommendations, str(output_dir))
    assert os.path.exists(filepath)
    
    # Test with invalid data types
    invalid_recommendations = [{
        'ticker': 'AAPL',
        'score': 'invalid',  # Should be number
        'price': 'not a number'
    }]
    filepath = save_recommendations_to_csv(invalid_recommendations, str(output_dir))
    assert os.path.exists(filepath)
    
    # Test with very large dataset
    large_recommendations = sample_recommendations * 1000  # 2000 recommendations
    filepath = save_recommendations_to_csv(large_recommendations, str(output_dir))
    assert os.path.exists(filepath)
    
    # An invalid directory should be handled gracefully by the implementation
    # and should log an appropriate error message without crashing
    save_recommendations_to_csv(sample_recommendations, '/invalid/dir')

def test_archive_and_cleanup_files(tmp_path):
    """Test file archiving and cleanup."""
    output_dir = tmp_path / 'output'
    archive_dir = tmp_path / 'archive'
    os.makedirs(output_dir)
    
    # Create test files
    test_files = {
        'test.csv': 'test data',
        'test.json': '{"key": "value"}',
        'test.txt': 'text data'  # Non-archived file type
    }
    for filename, content in test_files.items():
        (output_dir / filename).write_text(content)
    
    # Test archiving at 5 PM MT
    with patch('src.utils.common.get_current_time') as mock_time:
        mock_time.return_value = datetime(2024, 4, 9, 17, 0, tzinfo=pytz.timezone('America/Denver'))
        archive_and_cleanup_files(str(output_dir), str(archive_dir))
        assert os.path.exists(archive_dir)
        date_folder = os.path.join(archive_dir, '20240409')
        assert os.path.exists(date_folder)
        assert os.path.exists(os.path.join(date_folder, 'test.csv'))
        assert os.path.exists(os.path.join(date_folder, 'test.json'))
        assert not os.path.exists(os.path.join(date_folder, 'test.txt'))  # Not archived
        
    # Test cleanup at 7 AM MT
    with patch('src.utils.common.get_current_time') as mock_time:
        mock_time.return_value = datetime(2024, 4, 9, 7, 0, tzinfo=pytz.timezone('America/Denver'))
        archive_and_cleanup_files(str(output_dir), str(archive_dir))
        assert not os.path.exists(os.path.join(output_dir, 'test.csv'))
        assert not os.path.exists(os.path.join(output_dir, 'test.json'))
        assert os.path.exists(os.path.join(output_dir, 'test.txt'))  # Not cleaned up
    
    # Test with non-existent directories (should not raise error)
    try:
        archive_and_cleanup_files('/nonexistent/output', '/nonexistent/archive')
    except Exception as e:
        pytest.fail(f"archive_and_cleanup_files raised an unexpected exception: {e}")
    
    # Test with read-only files
    read_only_file = output_dir / 'readonly.csv'
    read_only_file.write_text('test data')
    read_only_file.chmod(0o444)  # Read-only
    with patch('src.utils.common.get_current_time') as mock_time:
        mock_time.return_value = datetime(2024, 4, 9, 7, 0, tzinfo=pytz.timezone('America/Denver'))
        archive_and_cleanup_files(str(output_dir), str(archive_dir))
        # File should still exist due to read-only status
        assert os.path.exists(read_only_file) 
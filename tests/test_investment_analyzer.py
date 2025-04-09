import os
import pytest
import pandas as pd
import json
import numpy as np
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from unittest.mock import MagicMock
from unittest.mock import patch

@pytest.mark.critical
class TestInvestmentAnalyzer:
    """Critical tests for the Investment Analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an instance of InvestmentAnalyzer."""
        return InvestmentAnalyzer()
    
    @pytest.fixture
    def sample_securities_data(self):
        """Create sample securities data for testing."""
        return pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'Name': ['Apple Inc', 'Alphabet Inc', 'Microsoft Corp'],
            'Last': [150.0, 2800.0, 300.0],
            'Volume': [1000000, 500000, 750000],
            '50D Avg Vol': [900000, 450000, 800000],
            '200D Avg Vol': [850000, 400000, 700000],
            '50D MA': [145.0, 2750.0, 295.0],
            '200D MA': [140.0, 2700.0, 290.0],
            'rsi': [65.0, 45.0, 70.0],
            'Beta': [1.2, 1.1, 1.0],
            '14D MACD': [2.5, -1.0, 1.5],
            '14D MACD_Signal': [2.0, -1.5, 1.0],
            '% Insider': ['5.2%', '4.8%', '3.5%'],
            'Div Yield(a)': ['0.5%', '0%', '1.8%'],
            'Market Cap': ['2.5T', '1.8T', '2.1T'],
            'Price Vol': [1.2, 1.1, 1.3],
            'P/E fwd': [25.5, 22.3, 28.4],
            'Price/Book': [35.2, 5.4, 12.8],
            'Debt/Equity': [1.5, 0.8, 0.6]
        })
    
    def test_clean_numeric(self, analyzer):
        """Test the _clean_numeric method."""
        assert analyzer._clean_numeric('123.45') == 123.45
        assert analyzer._clean_numeric('1.2M') == 1.2
        assert analyzer._clean_numeric('N/A') == 0.0
        assert analyzer._clean_numeric(None) == 0.0
        assert analyzer._clean_numeric('') == 0.0
        assert analyzer._clean_numeric('2.5T') == 2.5
        assert analyzer._clean_numeric('-123.45') == -123.45
    
    def test_load_and_prepare_securities(self, analyzer, sample_securities_data, tmp_path):
        """Test the load_and_prepare_securities method."""
        # Create a temporary CSV file
        test_csv = tmp_path / "test_securities.csv"
        sample_securities_data.to_csv(test_csv, index=False)
        
        # Set the input file to our test file
        analyzer.input_file = str(test_csv)
        
        # Test the method
        result = analyzer.load_and_prepare_securities()
        
        # Verify the result
        assert not result.empty
        assert 'volume_change_50d' in result.columns
        assert 'volume_change_200d' in result.columns
        assert 'price_momentum_50d' in result.columns
        assert 'price_momentum_200d' in result.columns
        
        # Test calculated metrics
        assert result.loc[0, 'volume_change_50d'] == pytest.approx((1000000 / 900000) - 1)
        assert result.loc[0, 'price_momentum_50d'] == pytest.approx((150.0 / 145.0) - 1)
    
    def test_calculate_price_movement_potential(self, analyzer, sample_securities_data):
        """Test the _calculate_price_movement_potential method."""
        result = analyzer._calculate_price_movement_potential(sample_securities_data)
        
        # Verify the result has the expected columns
        assert 'price_movement_potential' in result.columns
        assert 'rsi_potential' not in result.columns  # Should be dropped
        assert 'macd_potential' not in result.columns  # Should be dropped
        assert 'volume_potential' not in result.columns  # Should be dropped
        
        # Verify the values are within expected range
        assert all(0 <= x <= 1 for x in result['price_movement_potential'])
    
    def test_identify_potential_movers(self, analyzer):
        """Test the identify_potential_movers method."""
        # Create a sample DataFrame with test data
        test_data = pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'Name': ['Apple Inc', 'Alphabet Inc', 'Microsoft Corp'],
            'Last': [150.0, 2800.0, 300.0],
            'Volume': [1000000, 500000, 750000],
            'RSI': [65, 45, 55],
            'MACD': [2.5, -1.0, 1.5],
            'Volume_MA': [900000, 600000, 700000]
        })
        
        # Mock the load_and_prepare_securities method using patch
        with patch.object(analyzer, 'load_and_prepare_securities', return_value=test_data):
            # Call the method
            result = analyzer.identify_potential_movers()
            
            # Assertions
            assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
            assert 'price_movement_potential' in result.columns, "Result should have price_movement_potential column"
            assert 'analysis_timestamp' in result.columns, "Result should have analysis_timestamp column"
            assert all(0 <= score <= 1 for score in result['price_movement_potential']), "All scores should be between 0 and 1"
    
    def test_error_handling(self, analyzer):
        """Test error handling."""
        # Test with non-existent file
        analyzer.input_file = "non_existent_file.csv"
        result = analyzer.load_and_prepare_securities()
        assert result.empty

        # Test with invalid data
        invalid_df = pd.DataFrame({
            'Symbol': ['TEST'],
            'Last': ['invalid'],
            'Volume': ['invalid']
        })
        result = analyzer._calculate_price_movement_potential(invalid_df)
        assert not result.empty  # Should return original DataFrame without crashing 
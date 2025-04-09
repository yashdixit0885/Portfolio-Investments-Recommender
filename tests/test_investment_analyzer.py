import os
import pytest
import pandas as pd
import json
import numpy as np
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from unittest.mock import MagicMock, patch

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
            'RSI': [65.0, 45.0, 70.0],
            'Beta': [1.2, 1.1, 1.0],
            'MACD': [2.5, -1.0, 1.5],
            'MACD Signal': [2.0, -1.5, 1.0],
            '% Insider': ['5.2%', '4.8%', '3.5%'],
            'Div Yield(a)': ['0.5%', '0%', '1.8%'],
            'Market Cap': ['2.5T', '1.8T', '2.1T'],
            'Price Vol': [1.2, 1.1, 1.3],
            'P/E fwd': [25.5, 22.3, 28.4],
            'Price/Book': [35.2, 5.4, 12.8],
            'Debt/Equity': [1.5, 0.8, 0.6],
            'ATR': [2.5, 3.0, 2.0],
            'BB Width': [0.15, 0.12, 0.18],
            'Stoch %K': [75.0, 45.0, 80.0],
            'Stoch %D': [70.0, 40.0, 75.0],
            'ADX': [25.0, 30.0, 20.0],
            'OBV': [1000000, 500000, 750000],
            'VWAP': [149.0, 2795.0, 298.0]
        })
    
    def test_clean_numeric(self, analyzer):
        """Test the _clean_numeric method."""
        assert analyzer._clean_numeric('123.45') == 123.45
        assert analyzer._clean_numeric('1.2M') == 1200000.0
        assert analyzer._clean_numeric('N/A') == 0.0
        assert analyzer._clean_numeric(None) == 0.0
        assert analyzer._clean_numeric('') == 0.0
        assert analyzer._clean_numeric('2.5T') == 2500000000000.0
        assert analyzer._clean_numeric('-123.45') == -123.45
        assert analyzer._clean_numeric('5.2%') == 0.052
    
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
        
        # Verify column mappings
        assert 'Ticker' in result.columns  # Renamed from Symbol
        assert 'Price' in result.columns   # Renamed from Last
        
        # Verify percentage columns are properly converted
        assert result.loc[0, 'Inst Own %'] == pytest.approx(0.052)  # From 5.2%
    
    def test_calculate_price_movement_potential(self, analyzer, sample_securities_data):
        """Test the _calculate_price_movement_potential method."""
        # Prepare the data first
        prepared_data = analyzer.load_and_prepare_securities(sample_securities_data)
        result = analyzer._calculate_price_movement_potential(prepared_data)
        
        # Verify the result has the expected columns
        assert 'price_movement_potential' in result.columns
        
        # Verify the scoring components
        assert all(0 <= x <= 1 for x in result['price_movement_potential'])
        
        # Test specific scoring components
        # Price Action (30%)
        assert all(0 <= x <= 0.3 for x in result['price_momentum_score'])
        
        # Volume Analysis (25%)
        assert all(0 <= x <= 0.25 for x in result['volume_score'])
        
        # Technical Indicators (25%)
        assert all(0 <= x <= 0.25 for x in result['technical_score'])
        
        # Market Context (20%)
        assert all(0 <= x <= 0.20 for x in result['market_score'])
        
        # Verify AAPL has higher score than MSFT (due to higher RSI and volume)
        aapl_idx = result[result['Ticker'] == 'AAPL'].index[0]
        msft_idx = result[result['Ticker'] == 'MSFT'].index[0]
        assert result.loc[aapl_idx, 'price_movement_potential'] > result.loc[msft_idx, 'price_movement_potential']
    
    def test_calculate_technical_indicators(self, analyzer):
        """Test the _calculate_technical_indicators method."""
        # Create sample historical data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        history = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 30),
            'High': np.random.uniform(200, 300, 30),
            'Low': np.random.uniform(50, 100, 30),
            'Close': np.random.uniform(100, 200, 30),
            'Volume': np.random.uniform(1000000, 2000000, 30)
        }, index=dates)
        
        # Calculate indicators
        indicators = analyzer._calculate_technical_indicators(history)
        
        # Verify all expected indicators are present
        expected_indicators = [
            'atr', 'bb_width', 'rsi', 'stoch_k', 'stoch_d',
            'adx', 'obv', 'vwap', 'macd', 'macd_signal', 'macd_hist'
        ]
        for indicator in expected_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], float)

        # Verify values are within expected ranges
        assert 0 <= indicators['rsi'] <= 100, "RSI should be between 0 and 100"
        assert 0 <= indicators['stoch_k'] <= 100, "Stochastic %K should be between 0 and 100"
        assert 0 <= indicators['stoch_d'] <= 100, "Stochastic %D should be between 0 and 100"
        assert indicators['atr'] >= 0, "ATR should be non-negative"
        assert indicators['bb_width'] >= 0, "Bollinger Band width should be non-negative"
        assert indicators['adx'] >= 0, "ADX should be non-negative"
    
    def test_identify_potential_movers(self, analyzer):
        """Test the identify_potential_movers method."""
        # Create a sample DataFrame with test data
        test_data = pd.DataFrame({
            'Ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'Price': [150.0, 2800.0, 300.0],
            'Volume': [1000000, 500000, 750000],
            'RSI': [65, 45, 55],
            'MACD': [2.5, -1.0, 1.5],
            'Beta': [1.2, 1.1, 1.0],
            'Price Vol': [1.2, 1.1, 1.3],
            '50D MA': [145.0, 2750.0, 295.0],
            '200D MA': [140.0, 2700.0, 290.0]
        })
        
        # Mock the load_and_prepare_securities method
        with patch.object(analyzer, 'load_and_prepare_securities', return_value=test_data):
            # Call the method
            result = analyzer.identify_potential_movers()
            
            # Assertions
            assert isinstance(result, pd.DataFrame)
            assert 'price_movement_potential' in result.columns
            assert 'analysis_timestamp' in result.columns
            assert all(0 <= score <= 1 for score in result['price_movement_potential'])
            
            # Verify sorting (highest potential first)
            assert result['price_movement_potential'].is_monotonic_decreasing
    
    def test_error_handling(self, analyzer):
        """Test error handling in various scenarios."""
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
        
        # Test with missing columns
        missing_cols_df = pd.DataFrame({
            'Symbol': ['TEST'],
            'Last': [100.0]
        })
        result = analyzer._calculate_price_movement_potential(missing_cols_df)
        assert 'price_movement_potential' in result.columns
        assert result.loc[0, 'price_movement_potential'] == 0.0  # Should default to 0
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = analyzer._calculate_price_movement_potential(empty_df)
        assert result.empty

    def test_fetch_yf_data(self, analyzer):
        """Test the _fetch_yf_data method."""
        # Mock yfinance Ticker
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'regularMarketPrice': 150.0,
            'regularMarketVolume': 1000000,
            'marketCap': 2500000000000
        }
        mock_ticker.history.return_value = pd.DataFrame({
            'Open': [100.0],
            'High': [200.0],
            'Low': [50.0],
            'Close': [150.0],
            'Volume': [1000000]
        }, index=[pd.Timestamp.now()])

        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = analyzer._fetch_yf_data('AAPL')
            
            assert isinstance(result, dict)
            assert 'info' in result
            assert 'history' in result
            assert not result['history'].empty
            assert result['info']['regularMarketPrice'] == 150.0

    def test_calculate_opportunity_score(self, analyzer):
        """Test the _calculate_opportunity_score method."""
        # Test with complete metrics
        metrics = {
            'price_momentum_50d': 0.1,
            'price_momentum_200d': 0.2,
            'volume_change_50d': 0.15,
            'volume_change_200d': 0.25,
            'rsi': 55.0,
            'macd': 2.5,
            'pe_ratio': 20.0,
            'price_to_book': 2.5
        }
        score = analyzer._calculate_opportunity_score(metrics)
        assert 0 <= score <= 1

        # Test with missing metrics
        incomplete_metrics = {
            'price_momentum_50d': 0.1,
            'volume_change_50d': 0.15
        }
        score = analyzer._calculate_opportunity_score(incomplete_metrics)
        assert 0 <= score <= 1

        # Test with empty metrics
        score = analyzer._calculate_opportunity_score({})
        assert score == 0.0

    def test_save_potential_securities(self, analyzer, tmp_path):
        """Test the save_potential_securities method."""
        # Set up test data
        analyzer.output_file = str(tmp_path / "test_output.json")
        test_securities = [
            {'Ticker': 'AAPL', 'Score': 0.8},
            {'Ticker': 'GOOGL', 'Score': 0.7}
        ]

        # Test successful save
        result = analyzer.save_potential_securities(test_securities)
        assert result is True
        assert os.path.exists(analyzer.output_file)

        # Verify file contents
        with open(analyzer.output_file, 'r') as f:
            saved_data = json.load(f)
            assert len(saved_data) == 2
            assert saved_data[0]['Ticker'] == 'AAPL'

        # Test with empty list
        result = analyzer.save_potential_securities([])
        assert result is False

    def test_run_analysis(self, analyzer):
        """Test the run_analysis method."""
        # Mock identify_potential_movers to return test data
        test_data = pd.DataFrame({
            'Ticker': ['AAPL', 'GOOGL'],
            'price_movement_potential': [0.8, 0.7]
        })
        
        with patch.object(analyzer, 'identify_potential_movers', return_value=test_data), \
             patch.object(analyzer, 'save_potential_securities', return_value=True):
            result = analyzer.run_analysis()
            assert result is True

        # Test with empty results
        with patch.object(analyzer, 'identify_potential_movers', return_value=pd.DataFrame()), \
             patch.object(analyzer, 'save_potential_securities', return_value=False):
            result = analyzer.run_analysis()
            assert result is False

    def test_error_handling_extended(self, analyzer):
        """Test additional error handling scenarios."""
        # Test _fetch_yf_data error handling
        with patch('yfinance.Ticker', side_effect=Exception("API Error")):
            result = analyzer._fetch_yf_data('AAPL')
            assert isinstance(result, dict)
            assert result['info'] == {}
            assert isinstance(result['history'], pd.DataFrame)
            assert result['history'].empty

        # Test _calculate_technical_indicators error handling
        with patch('pandas.Series.rolling', side_effect=Exception("Calculation Error")):
            result = analyzer._calculate_technical_indicators(pd.DataFrame())
            assert isinstance(result, dict)
            assert all(v == 0.0 for v in result.values())

        # Test _calculate_opportunity_score error handling
        with patch('pandas.Series.mean', side_effect=Exception("Mean Error")):
            result = analyzer._calculate_opportunity_score(pd.DataFrame())
            assert isinstance(result, float)
            assert result == 0.0 
import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.research_analyst.research_analyzer import ResearchAnalyzer

@pytest.mark.critical
class TestResearchAnalyzer:
    """Critical tests for the Research Analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an instance of ResearchAnalyzer."""
        return ResearchAnalyzer()
    
    @pytest.fixture
    def sample_opportunity(self):
        """Create a sample opportunity dictionary."""
        return {
            'Symbol': 'AAPL',
            'Name': 'Apple Inc',
            'Last': 150.0,
            'Industry': 'Technology',
            'Market Cap': '2.5T',
            'Volume': 1000000,
            '50D Avg Vol': 900000,
            '200D Avg Vol': 850000,
            '50D MA': 145.0,
            '200D MA': 140.0,
            'Beta': 1.2,
            '14D Rel Str': 1.05
        }
    
    @pytest.fixture
    def sample_potential_securities(self):
        """Create sample potential securities for testing."""
        return [
            {
                'Ticker': 'AAPL',
                'Name': 'Apple Inc',
                'Last': 150.0,
                'Industry': 'Technology',
                'Market Cap': 2500000000000,
                'Volume': 1000000,
                'price_momentum_50d': 0.05,
                'price_momentum_200d': 0.07,
                'volume_change_50d': 0.11,
                'volume_change_200d': 0.18,
                'rsi': 0.55,
                'Beta': 1.2,
                'movement_potential_score': 0.85
            },
            {
                'Ticker': 'MSFT',
                'Name': 'Microsoft Corp',
                'Last': 280.0,
                'Industry': 'Technology',
                'Market Cap': 2000000000000,
                'Volume': 800000,
                'price_momentum_50d': 0.03,
                'price_momentum_200d': 0.04,
                'volume_change_50d': 0.07,
                'volume_change_200d': 0.14,
                'rsi': 0.60,
                'Beta': 0.9,
                'movement_potential_score': 0.75
            }
        ]
    
    @pytest.fixture
    def mock_yf_data(self):
        """Create mock yfinance data for testing."""
        return {
            'info': {
                'regularMarketPrice': 150.0,
                'debtToEquity': 300.0,  # High debt
                'profitMargins': -0.1,  # Negative margins
                'operatingMargins': -0.2,  # Negative margins
                'currentRatio': 0.8,  # Poor liquidity
                'beta': 2.0,  # High beta
                'averageVolume': 50000,  # Low volume
                'sector': 'Technology',  # High risk sector
                'earningsGrowth': -0.1,  # Negative growth
                'revenueGrowth': -0.2,  # Negative growth
                'heldPercentInstitutions': 0.1,  # Low institutional ownership
                'shortRatio': 15.0  # High short interest
            },
            'history': pd.DataFrame({
                'Close': [150.0, 140.0, 130.0, 120.0, 110.0, 100.0],  # Downtrend
                'High': [155.0, 145.0, 135.0, 125.0, 115.0, 105.0],
                'Low': [145.0, 135.0, 125.0, 115.0, 105.0, 95.0],
                'Volume': [50000, 45000, 40000, 35000, 30000, 25000]  # Declining volume
            })
        }

    def test_fetch_yf_data_error_handling(self, analyzer):
        """Test error handling in _fetch_yf_data method."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Test case 1: Empty info
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {}
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance
            
            result = analyzer._fetch_yf_data('INVALID')
            assert result['info'] == {}
            assert result['history'].empty
            
            # Test case 2: Exception during fetch
            mock_ticker.side_effect = Exception("API Error")
            result = analyzer._fetch_yf_data('ERROR')
            assert result['info'] == {}
            assert result['history'].empty
            
            # Test case 3: Cache hit
            mock_ticker.side_effect = None  # Reset the side effect
            mock_ticker.return_value = mock_ticker_instance  # Reset the return value
            cache_data = {'info': {'test': 'data'}, 'history': pd.DataFrame()}
            analyzer.data_cache[f"CACHED_data"] = cache_data
            result = analyzer._fetch_yf_data('CACHED')
            assert result['info'] == cache_data['info']
            assert result['history'].equals(cache_data['history'])

    def test_calculate_volatility_metrics_edge_cases(self, analyzer):
        """Test edge cases in _calculate_volatility_metrics method."""
        # Test case 1: Empty DataFrame
        result = analyzer._calculate_volatility_metrics(pd.DataFrame())
        assert all(v is None for v in result.values())
        
        # Test case 2: Single row DataFrame
        single_row = pd.DataFrame({'Close': [100.0]})
        result = analyzer._calculate_volatility_metrics(single_row)
        assert all(v is None for v in result.values())
        
        # Test case 3: Missing Close column
        invalid_df = pd.DataFrame({'Open': [100.0, 101.0]})
        result = analyzer._calculate_volatility_metrics(invalid_df)
        assert all(v is None for v in result.values())
        
        # Test case 4: All identical prices
        flat_prices = pd.DataFrame({'Close': [100.0] * 10})
        result = analyzer._calculate_volatility_metrics(flat_prices)
        assert result['std_dev'] == 0.0
        assert result['max_drawdown'] == 0.0
        assert result['upside_vol'] == 0.0
        assert result['downside_vol'] == 0.0

    def test_calculate_risk_score_edge_cases(self, analyzer, mock_yf_data):
        """Test edge cases in _calculate_risk_score method."""
        with patch.object(analyzer, '_fetch_yf_data', return_value=mock_yf_data):
            # Test case 1: Missing required fields
            minimal_security = {'Ticker': 'TEST'}
            result = analyzer._calculate_risk_score(minimal_security)
            assert isinstance(result, int)
            assert 0 <= result <= 100
            
            # Test case 2: Extreme values
            extreme_security = {
                'Ticker': 'EXTREME',
                'Beta': 5.0,
                'rsi': 0.95,
                'Market Cap': 100000,  # Extremely small
                'Volume': 10,  # Extremely low
                'price_momentum_50d': 2.0,  # Extreme momentum
                'price_momentum_200d': 2.0,  # Extreme momentum
                'volume_change_50d': 5.0,  # Extreme volume change
                'volume_change_200d': 5.0   # Extreme volume change
            }
            result = analyzer._calculate_risk_score(extreme_security)
            assert result > 70  # Should be high risk
            
            # Test case 3: All fields null/zero
            null_security = {
                'Ticker': 'NULL',
                'Beta': 0,
                'rsi': 0,
                'Market Cap': 0,
                'Volume': 0,
                'price_momentum_50d': 0,
                'price_momentum_200d': 0
            }
            result = analyzer._calculate_risk_score(null_security)
            assert isinstance(result, int)
            assert 0 <= result <= 100

    def test_run_risk_analysis_error_handling(self, analyzer, tmp_path):
        """Test error handling in run_risk_analysis method."""
        # Test case 1: Input file doesn't exist
        analyzer.input_file = 'nonexistent.json'
        result = analyzer.run_risk_analysis()
        assert result is False
        
        # Test case 2: Invalid JSON in input file
        input_file = tmp_path / "invalid.json"
        with open(input_file, 'w') as f:
            f.write("invalid json")
        analyzer.input_file = str(input_file)
        result = analyzer.run_risk_analysis()
        assert result is False
        
        # Test case 3: Empty securities list
        input_file = tmp_path / "empty.json"
        with open(input_file, 'w') as f:
            json.dump([], f)
        analyzer.input_file = str(input_file)
        result = analyzer.run_risk_analysis()
        assert result is False
        
        # Test case 4: Output directory doesn't exist
        analyzer.output_file = '/nonexistent/dir/output.json'
        result = analyzer.run_risk_analysis()
        assert result is False

    def test_fetch_yf_data(self, analyzer, mock_yf_data):
        """Test the _fetch_yf_data method."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Configure the mock
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = mock_yf_data['info']
            mock_ticker_instance.history.return_value = mock_yf_data['history']
            mock_ticker.return_value = mock_ticker_instance
            
            # Call the method
            result = analyzer._fetch_yf_data('AAPL')
            
            # Check the result
            assert isinstance(result, dict)
            assert 'info' in result
            assert 'history' in result
            assert result['info'] == mock_yf_data['info']
            assert isinstance(result['history'], pd.DataFrame)
    
    def test_calculate_volatility_metrics(self, analyzer):
        """Test the _calculate_volatility_metrics method."""
        # Create a sample history DataFrame with price movements
        history = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0]
        })
        
        # Call the method
        result = analyzer._calculate_volatility_metrics(history)
        
        # Check the result
        assert isinstance(result, dict)
        assert all(key in result for key in ['std_dev', 'max_drawdown', 'upside_vol', 'downside_vol'])
        assert all(isinstance(result[key], (float, type(None))) for key in result)
        assert result['std_dev'] is not None
        assert result['max_drawdown'] is not None
        assert result['upside_vol'] is not None
        assert result['downside_vol'] is not None
    
    def test_calculate_risk_score(self, analyzer, sample_potential_securities, mock_yf_data):
        """Test the _calculate_risk_score method with new risk factors."""
        with patch.object(analyzer, '_fetch_yf_data', return_value=mock_yf_data):
            # Call the method
            security = sample_potential_securities[0]
            result = analyzer._calculate_risk_score(security)
            
            # Check the result
            assert isinstance(result, int)
            assert 0 <= result <= 100
            
            # Test with different risk profiles
            # High risk profile
            high_risk_security = security.copy()
            high_risk_security['Beta'] = 2.0
            high_risk_security['rsi'] = 0.8  # Extreme RSI
            high_risk_result = analyzer._calculate_risk_score(high_risk_security)
            assert high_risk_result > result
            
            # Low risk profile
            low_risk_security = security.copy()
            low_risk_security['Beta'] = 0.5
            low_risk_security['rsi'] = 0.5  # Neutral RSI
            low_risk_result = analyzer._calculate_risk_score(low_risk_security)
            assert low_risk_result < result
    
    def test_run_risk_analysis(self, analyzer, sample_potential_securities, tmp_path):
        """Test the run_risk_analysis method."""
        # Create a temporary input file
        input_file = tmp_path / "high_potential_securities.json"
        with open(input_file, 'w') as f:
            json.dump(sample_potential_securities, f)
        
        # Create a temporary output file
        output_file = tmp_path / "risk_scored_securities.json"
        
        # Monkeypatch the file paths
        analyzer.input_file = str(input_file)
        analyzer.output_file = str(output_file)
        
        # Mock the _calculate_risk_score method
        with patch.object(analyzer, '_calculate_risk_score', return_value=50):
            # Call the method
            result = analyzer.run_risk_analysis()
            
            # Check the result
            assert isinstance(result, dict)
            assert len(result) == 2  # Should have results for both AAPL and MSFT
            assert 'AAPL' in result
            assert 'MSFT' in result
            assert result['AAPL']['risk_score'] == 50
            assert result['MSFT']['risk_score'] == 50 
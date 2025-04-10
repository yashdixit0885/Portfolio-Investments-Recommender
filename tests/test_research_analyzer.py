import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.database.db_manager import DatabaseManager

@pytest.mark.critical
class TestResearchAnalyzer:
    """Critical tests for the Research Analyzer."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseManager."""
        mock = MagicMock(spec=DatabaseManager)
        mock.get_all_securities.return_value = [
            {'ticker': 'AAPL', 'name': 'Apple Inc', 'industry': 'Tech', 'beta': 1.2, 'inst_own_pct': 0.6},
            {'ticker': 'MSFT', 'name': 'Microsoft', 'industry': 'Tech', 'beta': 0.9, 'inst_own_pct': 0.7}
        ]
        # Sample history for risk calculation - USE LOWERCASE COLUMNS
        mock.get_historical_data.return_value = pd.DataFrame({
            'close': [100, 110, 105, 115, 120], # ensure variance for volatility > 0
            'volume': [1e6, 1.1e6, 1.05e6, 1.15e6, 1.2e6]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))
        mock.insert_analysis_result.return_value = True
        return mock

    @pytest.fixture
    def analyzer(self, mock_db):
        """Create an instance of ResearchAnalyzer with a mock DB."""
        return ResearchAnalyzer(db=mock_db)
    
    def test_calculate_risk_metrics(self, analyzer, mock_db):
        """Test the _calculate_risk_metrics method."""
        # Get sample historical data from the mock db
        hist_data = mock_db.get_historical_data('AAPL')
        security_info = {'beta': 1.2, 'inst_own_pct': 0.6} # Sample info
        
        metrics = analyzer._calculate_risk_metrics(hist_data, security_info)
        
        assert isinstance(metrics, dict)
        assert 'risk_score' in metrics
        assert 'volatility' in metrics
        assert 'max_drawdown' in metrics
        assert 'beta' in metrics
        assert 'avg_volume' in metrics
        assert 'volume_volatility' in metrics
        assert 'institutional_ownership' in metrics
        assert 'timestamp' in metrics
        
        assert isinstance(metrics['risk_score'], (int, float))
        assert 0 <= metrics['risk_score'] <= 100
        # Check volatility is non-negative (might be 0 if variance is tiny)
        assert metrics['volatility'] >= 0 # Based on sample data 
        assert metrics['max_drawdown'] <= 0 # Max drawdown is non-positive

    def test_calculate_risk_metrics_edge_cases(self, analyzer):
        """Test edge cases for _calculate_risk_metrics."""
        security_info = {'beta': 1.0, 'inst_own_pct': 0.5}

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        metrics = analyzer._calculate_risk_metrics(empty_df, security_info)
        assert metrics['risk_score'] > 0 # Should assign some default risk
        assert metrics['volatility'] == 0
        assert metrics['max_drawdown'] == 0

        # Test with insufficient data (less than 2 periods)
        short_df = pd.DataFrame({'close': [100]})
        metrics = analyzer._calculate_risk_metrics(short_df, security_info)
        assert metrics['volatility'] == 0

        # Test with missing 'close' or 'volume'
        missing_close_df = pd.DataFrame({'volume': [1e6, 1.1e6]})
        metrics = analyzer._calculate_risk_metrics(missing_close_df, security_info)
        assert metrics['volatility'] == 0 # Calculation depends on close
        assert metrics['volume_volatility'] >= 0 

        missing_volume_df = pd.DataFrame({'close': [100, 110]})
        metrics = analyzer._calculate_risk_metrics(missing_volume_df, security_info)
        assert metrics['volatility'] >= 0 
        assert metrics['volume_volatility'] == 0 # Calculation depends on volume

    def test_run_risk_analysis(self, analyzer, mock_db):
        """Test the main run_risk_analysis method."""
        # Reset mocks before call
        mock_db.reset_mock()
        # Pass a list of tickers
        tickers = ['AAPL', 'MSFT']
        results = analyzer.run_risk_analysis(tickers)
        
        assert isinstance(results, dict)
        assert 'AAPL' in results
        assert 'MSFT' in results
        assert 'risk_score' in results['AAPL']
        
        # Verify database interactions
        assert mock_db.get_historical_data.call_count == 2 
        mock_db.get_historical_data.assert_any_call('AAPL')
        mock_db.get_historical_data.assert_any_call('MSFT')
        assert mock_db.insert_analysis_result.call_count == 2 
        
        # Define expected metrics structure for assertion
        from unittest.mock import ANY 
        # Check the call for AAPL explicitly
        aapl_call_args = None
        msft_call_args = None
        for call_args in mock_db.insert_analysis_result.call_args_list:
            args, kwargs = call_args
            if kwargs.get('ticker') == 'AAPL':
                 aapl_call_args = kwargs
            elif kwargs.get('ticker') == 'MSFT':
                 msft_call_args = kwargs

        assert aapl_call_args is not None
        assert msft_call_args is not None

        # Assert structure and types for AAPL call
        assert aapl_call_args['ticker'] == 'AAPL'
        assert aapl_call_args['analysis_type'] == 'risk'
        assert isinstance(aapl_call_args['score'], float)
        assert isinstance(aapl_call_args['metrics'], dict)
        # Check specific keys in metrics
        metrics = aapl_call_args['metrics']
        assert 'risk_score' in metrics
        assert 'volatility' in metrics
        assert 'max_drawdown' in metrics
        assert 'beta' in metrics
        assert 'avg_volume' in metrics
        assert 'volume_volatility' in metrics
        assert 'institutional_ownership' in metrics
        assert 'timestamp' in metrics
        # Verify score approx matches the one in metrics
        assert metrics['risk_score'] == pytest.approx(aapl_call_args['score'])

    def test_run_risk_analysis_error_handling(self, analyzer, mock_db):
        """Test error handling in run_risk_analysis method."""
        # Prepare a test ticker
        tickers = ['AAPL']
        
        # Test case 1: No matching security info in get_all_securities
        mock_db.get_all_securities.return_value = [
            {'ticker': 'MSFT', 'name': 'Microsoft'} # Different ticker than requested
        ]
        results = analyzer.run_risk_analysis(tickers)
        assert isinstance(results, dict)
        assert len(results) == 0  # Expect empty results if security info not found
        mock_db.get_historical_data.assert_not_called()
        mock_db.insert_analysis_result.assert_not_called()
        
        # Reset mocks for next test
        mock_db.reset_mock()
        
        # Test case 2: Security info exists but no historical data
        mock_db.get_all_securities.return_value = [
            {'ticker': 'AAPL', 'name': 'Apple Inc', 'industry': 'Tech'}
        ]
        mock_db.get_historical_data.return_value = None
        results = analyzer.run_risk_analysis(tickers)
        assert isinstance(results, dict)
        assert len(results) == 0  # Expect empty results if no historical data
        mock_db.get_historical_data.assert_called_once()
        mock_db.insert_analysis_result.assert_not_called()
        
        # Reset mocks for next test
        mock_db.reset_mock()
        
        # Test case 3: Both security info and historical data exist, but DB insert fails
        mock_db.get_all_securities.return_value = [
            {'ticker': 'AAPL', 'name': 'Apple Inc', 'industry': 'Tech'}
        ]
        mock_db.get_historical_data.return_value = pd.DataFrame({
            'close': [100, 110, 120],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_db.insert_analysis_result.return_value = False
        
        results = analyzer.run_risk_analysis(tickers)
        assert isinstance(results, dict)
        assert 'AAPL' not in results  # Should not include if insert fails
        mock_db.get_historical_data.assert_called_once()
        mock_db.insert_analysis_result.assert_called_once() 
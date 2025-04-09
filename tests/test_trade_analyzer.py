import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.trade_analyst.trade_analyzer import TradeAnalyzer

@pytest.mark.critical
class TestTradeAnalyzer:
    """Critical tests for the Trade Analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an instance of TradeAnalyzer."""
        return TradeAnalyzer()
    
    @pytest.fixture
    def sample_risk_scored_securities(self):
        """Create sample risk-scored securities for testing."""
        return {
            "analysis_timestamp": "2023-04-08T12:00:00",
            "risk_scored_securities": [
                {
                    'Symbol': 'AAPL',
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
                    'movement_potential_score': 0.85,
                    'risk_score': 30
                },
                {
                    'Symbol': 'MSFT',
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
                    'movement_potential_score': 0.75,
                    'risk_score': 25
                }
            ]
        }
    
    @pytest.fixture
    def mock_stock_data(self):
        """Create mock stock data for testing."""
        return pd.DataFrame({
            'Open': [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
            'High': [146.0, 147.0, 148.0, 149.0, 150.0, 151.0],
            'Low': [144.0, 145.0, 146.0, 147.0, 148.0, 149.0],
            'Close': [145.5, 146.5, 147.5, 148.5, 149.5, 150.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        })
    
    def test_get_stock_data(self, analyzer, mock_stock_data):
        """Test the _get_stock_data method."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Configure the mock
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = mock_stock_data
            mock_ticker.return_value = mock_ticker_instance
            
            # Call the method
            result = analyzer._get_stock_data('AAPL', period='1y')
            
            # Check the result
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'Close' in result.columns
            assert 'Volume' in result.columns
    
    def test_calculate_technical_indicators(self, analyzer, mock_stock_data):
        """Test the _calculate_technical_indicators method."""
        # Call the method
        result = analyzer._calculate_technical_indicators(mock_stock_data)
        
        # Check the result
        assert isinstance(result, dict)
        assert 'sma_20' in result
        assert 'sma_50' in result
        assert 'sma_200' in result
        assert 'ema_5' in result
        assert 'ema_10' in result
        assert 'ema_20' in result
        assert 'ema_50' in result
        assert 'macd' in result
        assert 'macd_signal' in result
        assert 'macd_hist' in result
        assert 'rsi' in result
        assert 'stoch_k' in result
        assert 'stoch_d' in result
        assert 'bb_hband' in result
        assert 'bb_mavg' in result
        assert 'bb_lband' in result
        assert 'bb_width' in result
        assert 'current_price' in result
    
    def test_determine_signal_and_timeframe(self, analyzer):
        """Test the _determine_signal_and_timeframe method."""
        # Create sample indicators
        indicators = {
            'sma_20': 145.0,
            'sma_50': 140.0,
            'sma_200': 135.0,
            'ema_5': 150.0,
            'ema_10': 148.0,
            'ema_20': 146.0,
            'ema_50': 144.0,
            'macd': 2.0,
            'macd_signal': 1.0,
            'macd_hist': 1.0,
            'rsi': 65.0,
            'stoch_k': 75.0,
            'stoch_d': 70.0,
            'bb_hband': 155.0,
            'bb_mavg': 145.0,
            'bb_lband': 135.0,
            'bb_width': 0.1,
            'current_price': 150.0
        }
        
        # Call the method
        signal, timeframe, justification = analyzer._determine_signal_and_timeframe(indicators)
        
        # Check the result
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert timeframe in ['short_term', 'medium_term', 'long_term', 'N/A']
        assert isinstance(justification, list)
        assert len(justification) > 0
    
    def test_generate_trade_signals(self, analyzer, sample_risk_scored_securities, tmp_path):
        """Test the generate_trade_signals method."""
        # Create a temporary input file
        input_file = tmp_path / "risk_scored_securities.json"
        with open(input_file, 'w') as f:
            json.dump(sample_risk_scored_securities, f)
        
        # Create a temporary output file
        output_file = tmp_path / "Trade_Recommendations_latest.csv"
        
        # Monkeypatch the file paths
        analyzer.input_file = str(input_file)
        analyzer.output_file = str(output_file)
        
        # Mock the _get_stock_data method
        mock_stock_data = pd.DataFrame({
            'Open': [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
            'High': [146.0, 147.0, 148.0, 149.0, 150.0, 151.0],
            'Low': [144.0, 145.0, 146.0, 147.0, 148.0, 149.0],
            'Close': [145.5, 146.5, 147.5, 148.5, 149.5, 150.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        })
        
        with patch.object(analyzer, '_get_stock_data', return_value=mock_stock_data):
            # Call the method
            result = analyzer.generate_trade_signals()
            
            # Check the result
            assert result is True
            assert output_file.exists()
            
            # Check the content of the file
            df = pd.read_csv(output_file)
            assert not df.empty
            assert 'Ticker' in df.columns
            assert 'Signal' in df.columns
            assert 'Timeframe' in df.columns
            assert 'Risk Score' in df.columns
            assert 'Current Price' in df.columns
            assert 'Justification' in df.columns
            assert 'Analysis Timestamp' in df.columns 
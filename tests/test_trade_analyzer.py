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
            "risk_scored_securities": [
                {
                    'symbol': 'AAPL',
                    'name': 'Apple Inc',
                    'risk_score': 30
                },
                {
                    'symbol': 'MSFT',
                    'name': 'Microsoft Corp',
                    'risk_score': 25
                }
            ]
        }
    
    @pytest.fixture
    def mock_stock_data(self):
        """Create mock stock data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'Open': [145.0 + i for i in range(20)],
            'High': [146.0 + i for i in range(20)],
            'Low': [144.0 + i for i in range(20)],
            'Close': [145.5 + i for i in range(20)],
            'Volume': [1000000 + i*100000 for i in range(20)]
        }, index=dates)
        
        # Add calculated columns
        df['HL_Range'] = df['High'] - df['Low']
        df['OC_Range'] = df['Close'] - df['Open']
        df['Body_Size'] = abs(df['OC_Range'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Swing_High'] = [False] * 20
        df['Swing_Low'] = [False] * 20
        df.loc[df.index[5], 'Swing_High'] = True
        df.loc[df.index[10], 'Swing_Low'] = True
        df.loc[df.index[15], 'Swing_High'] = True
        
        return df
    
    def test_get_stock_data(self, analyzer, mock_stock_data):
        """Test the _get_stock_data method."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Configure the mock
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = mock_stock_data
            mock_ticker.return_value = mock_ticker_instance
            
            # Call the method
            result = analyzer._get_stock_data('AAPL', period='20d')
            
            # Check the result
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'Close' in result.columns
            assert 'Volume' in result.columns
            assert 'HL_Range' in result.columns
            assert 'OC_Range' in result.columns
            assert 'Body_Size' in result.columns
            assert 'Upper_Shadow' in result.columns
            assert 'Lower_Shadow' in result.columns
            assert 'Swing_High' in result.columns
            assert 'Swing_Low' in result.columns
    
    def test_calculate_technical_indicators(self, analyzer, mock_stock_data):
        """Test the _calculate_technical_indicators method."""
        # Call the method
        result = analyzer._calculate_technical_indicators(mock_stock_data)
        
        # Check the result
        assert isinstance(result, dict)
        
        # Trend Indicators
        assert 'sma_5' in result
        assert 'sma_10' in result
        assert 'sma_20' in result
        assert 'sma_50' in result
        assert 'ema_5' in result
        assert 'ema_10' in result
        assert 'ema_20' in result
        assert 'ema_50' in result
        assert 'macd' in result
        assert 'macd_signal' in result
        assert 'macd_hist' in result
        assert 'adx' in result
        assert '+di' in result
        assert '-di' in result
        
        # Momentum Indicators
        assert 'rsi' in result
        assert 'stoch_k' in result
        assert 'stoch_d' in result
        assert 'willr' in result
        
        # Volume Indicators
        assert 'obv' in result
        assert 'cmf' in result
        assert 'volume_sma_20' in result
        
        # Volatility Indicators
        assert 'bb_upper' in result
        assert 'bb_middle' in result
        assert 'bb_lower' in result
        assert 'bb_width' in result
        assert 'atr' in result
        
        # Swing Analysis
        assert 'swing_highs' in result
        assert 'swing_lows' in result
        assert 'trend_channel' in result
        assert 'swing_size' in result
        assert 'swing_direction' in result
        
        # Price and Trend
        assert 'current_price' in result
        assert 'recent_trend' in result
    
    def test_analyze_swings(self, analyzer, mock_stock_data):
        """Test the _analyze_swings method."""
        # Call the method
        result = analyzer._analyze_swings(mock_stock_data)
        
        # Check the result
        assert isinstance(result, dict)
        assert 'swing_highs' in result
        assert 'swing_lows' in result
        assert 'trend_channel' in result
        assert 'swing_size' in result
        assert 'swing_direction' in result
        assert isinstance(result['trend_channel'], dict)
        
        # If there are enough swing points, check trend channel details
        if len(mock_stock_data[mock_stock_data['Swing_High']]) >= 2 and len(mock_stock_data[mock_stock_data['Swing_Low']]) >= 2:
            assert 'upper_slope' in result['trend_channel']
            assert 'lower_slope' in result['trend_channel']
            assert 'is_uptrend' in result['trend_channel']
            assert 'is_downtrend' in result['trend_channel']
            assert isinstance(result['trend_channel']['upper_slope'], float)
            assert isinstance(result['trend_channel']['lower_slope'], float)
            assert isinstance(result['trend_channel']['is_uptrend'], bool)
            assert isinstance(result['trend_channel']['is_downtrend'], bool)
    
    def test_analyze_recent_trend(self, analyzer, mock_stock_data):
        """Test the _analyze_recent_trend method."""
        # Call the method
        result = analyzer._analyze_recent_trend(mock_stock_data)
        
        # Check the result
        assert isinstance(result, dict)
        assert 'trend_strength' in result
        assert 'trend_direction' in result
        assert 'price_action' in result
        assert result['trend_direction'] in ['up', 'down', 'neutral']
        assert isinstance(result['price_action'], list)
    
    def test_determine_signal_and_timeframe(self, analyzer):
        """Test the _determine_signal_and_timeframe method."""
        # Create sample indicators
        indicators = {
            'adx': 30.0,
            '+di': 25.0,
            '-di': 20.0,
            'ema_10': 150.0,
            'ema_20': 148.0,
            'ema_50': 144.0,
            'rsi': 65.0,
            'cmf': 0.2,
            'current_price': 150.0,
            'atr': 2.0,
            'swing_analysis': {
                'swing_size': 0.1,
                'swing_direction': 'up',
                'trend_channel': {
                    'is_uptrend': True,
                    'is_downtrend': False,
                    'upper_slope': 0.5,
                    'lower_slope': 0.3
                }
            }
        }
        
        # Call the method
        signal, timeframe, justification = analyzer._determine_signal_and_timeframe(indicators)
        
        # Check the result
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert timeframe in ['short_swing', 'medium_swing', 'long_swing']
        assert isinstance(justification, list)
        assert len(justification) > 0
        assert justification[0].startswith(('STRONG', 'MODERATE', 'WEAK', 'NEUTRAL'))
    
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
            'Open': [145.0 + i for i in range(20)],
            'High': [146.0 + i for i in range(20)],
            'Low': [144.0 + i for i in range(20)],
            'Close': [145.5 + i for i in range(20)],
            'Volume': [1000000 + i*100000 for i in range(20)],
            'HL_Range': [2.0] * 20,
            'OC_Range': [0.5] * 20,
            'Body_Size': [0.5] * 20,
            'Upper_Shadow': [0.5] * 20,
            'Lower_Shadow': [0.5] * 20,
            'Swing_High': [False] * 20,
            'Swing_Low': [False] * 20
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
            assert 'ticker' in df.columns
            assert 'name' in df.columns
            assert 'signal' in df.columns
            assert 'timeframe' in df.columns
            assert 'confidence' in df.columns
            assert 'price' in df.columns
            assert 'position_size' in df.columns
            assert 'justification' in df.columns 
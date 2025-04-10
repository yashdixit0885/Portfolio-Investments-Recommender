import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.trade_analyst.trade_analyzer import TradeAnalyzer
from src.database.db_manager import DatabaseManager
from datetime import datetime

@pytest.mark.critical
class TestTradeAnalyzer:
    """Critical tests for the Trade Analyzer."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock DatabaseManager."""
        mock = MagicMock(spec=DatabaseManager)
        mock.get_all_securities.return_value = [{'ticker': 'AAPL', 'name': 'Apple Inc'}, {'ticker': 'MSFT', 'name': 'Microsoft Corp'}]
        # Mock get_latest_analysis to return a sample result for any ticker/type
        mock.get_latest_analysis.return_value = {
            'ticker': 'AAPL', # Example ticker
            'analysis_type': 'trade_signal',
            'score': 0.75,
            'metrics': {'signal_score': 0.75, 'trend': 'bullish'}, # Sample metrics
            'timestamp': datetime.now().isoformat()
        }
        mock.get_historical_data.return_value = pd.DataFrame({'Close': [100, 110, 105]})
        mock.insert_analysis_result.return_value = True
        return mock

    @pytest.fixture
    def analyzer(self, mock_db):
        """Create an instance of TradeAnalyzer with a mock DB."""
        return TradeAnalyzer(db=mock_db)
    
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
        """Create mock stock data for testing - use lowercase column names."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'open': [145.0 + i for i in range(20)],
            'high': [146.0 + i for i in range(20)],
            'low': [144.0 + i for i in range(20)],
            'close': [145.5 + i for i in range(20)], # Lowercase
            'volume': [1000000 + i*100000 for i in range(20)]
        }, index=dates)
        
        # Add calculated columns (ensure consistency with lowercase)
        df['hl_range'] = df['high'] - df['low']
        df['oc_range'] = df['close'] - df['open']
        df['body_size'] = abs(df['oc_range'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['swing_high'] = [False] * 20
        df['swing_low'] = [False] * 20
        df.loc[df.index[5], 'swing_high'] = True
        df.loc[df.index[10], 'swing_low'] = True
        df.loc[df.index[15], 'swing_high'] = True
        
        return df
    
    def test_get_stock_data(self, analyzer, mock_stock_data):
        """Test the _get_stock_data method."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Configure the mock
            mock_ticker_instance = MagicMock()
            # Ensure the mock history itself has lowercase columns
            mock_ticker_instance.history.return_value = mock_stock_data 
            mock_ticker.return_value = mock_ticker_instance
            
            # Call the method
            result = analyzer._get_stock_data('AAPL', period='20d')
            
            # Check the result (should be lowercase)
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'close' in result.columns # Expect lowercase
            assert 'volume' in result.columns
            # Check calculated columns are also lowercase
            assert 'hl_range' in result.columns 
            assert 'oc_range' in result.columns
            assert 'body_size' in result.columns
            assert 'upper_shadow' in result.columns
            assert 'lower_shadow' in result.columns
            assert 'swing_high' in result.columns
            assert 'swing_low' in result.columns
    
    def test_calculate_technical_indicators(self, analyzer, mock_stock_data):
        """Test the _calculate_technical_indicators method."""
        # Call the method with lowercase data
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
        # Call the method with lowercase data
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
        if len(mock_stock_data[mock_stock_data['swing_high']]) >= 2 and len(mock_stock_data[mock_stock_data['swing_low']]) >= 2:
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
        # Call the method with lowercase data
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
        
        # Mock the database returns
        mock_security = {'ticker': 'AAPL', 'name': 'Apple Inc', 'price': 150.0}
        analyzer.db.get_security.return_value = mock_security
        
        # Mock historical data
        mock_historical_data = pd.DataFrame({
            'open': [145.0 + i for i in range(20)],
            'high': [146.0 + i for i in range(20)],
            'low': [144.0 + i for i in range(20)],
            'close': [145.5 + i for i in range(20)], 
            'volume': [1000000 + i*100000 for i in range(20)]
        })
        analyzer.db.get_historical_data.return_value = mock_historical_data
        
        # Mock analysis results
        mock_investment_analysis = {
            'score': 0.75,
            'metrics': {'trend': 'bullish', 'momentum': 'positive'}
        }
        mock_risk_analysis = {
            'risk_score': 0.3,
            'metrics': {'volatility': 0.2, 'beta': 1.1}
        }
        analyzer.db.get_latest_analysis.side_effect = lambda ticker, analysis_type: \
            mock_investment_analysis if analysis_type == 'investment_opportunity' else mock_risk_analysis
        
        # Mock successful DB insertion
        analyzer.db.insert_analysis_result.return_value = True
        
        # Create list of tickers to pass to the method
        tickers = ['AAPL', 'MSFT']
        
        # Mock the _generate_signals method to return a valid signal dictionary
        mock_signal = {
            'signal_score': 0.8,
            'signal': 'BUY',
            'timeframe': 'MEDIUM',
            'confidence': 'HIGH',
            'current_price': 155.0,
            'position_size': 1000,
            'justification': json.dumps(['HIGH confidence: Strong uptrend detected'])
        }
        with patch.object(analyzer, '_generate_signals', return_value=mock_signal):
            # Call the method with tickers
            result = analyzer.generate_trade_signals(tickers)
            
            # Check the result - should be a dictionary of signals
            assert isinstance(result, dict)
            assert 'AAPL' in result
            assert 'MSFT' in result
            assert result['AAPL']['signal_score'] == 0.8
            assert result['AAPL']['signal'] == 'BUY'
            
            # Verify DB calls were made
            assert analyzer.db.get_security.call_count == 2
            assert analyzer.db.get_historical_data.call_count == 2
            assert analyzer.db.get_latest_analysis.call_count == 4  # 2 tickers * 2 analysis types
            assert analyzer.db.insert_analysis_result.call_count == 2 
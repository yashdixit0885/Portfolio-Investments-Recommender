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
                'movement_potential_score': 0.85
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
                'movement_potential_score': 0.75
            }
        ]
    
    @pytest.fixture
    def mock_yf_data(self):
        """Create mock yfinance data for testing."""
        return {
            'info': {
                'regularMarketPrice': 150.0,
                'debtToEquity': 150.0,
                'profitMargins': 0.25,
                'beta': 1.2
            },
            'history': pd.DataFrame({
                'Close': [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
                'High': [146.0, 147.0, 148.0, 149.0, 150.0, 151.0],
                'Low': [144.0, 145.0, 146.0, 147.0, 148.0, 149.0],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
            })
        }
    
    def test_analyze_fundamentals(self, analyzer):
        """Test the analyze_fundamentals method."""
        # Test with a valid ticker
        result = analyzer.analyze_fundamentals('AAPL')
        
        assert isinstance(result, dict)
        assert 'pe_ratio' in result
        assert 'price_to_book' in result
        assert 'debt_to_equity' in result
        assert 'profit_margin' in result
        assert 'revenue_growth' in result
        assert 'earnings_growth' in result
        assert 'current_ratio' in result
        assert 'return_on_equity' in result
        assert 'beta' in result
        assert 'volatility' in result
        
        # Test with an invalid ticker
        result = analyzer.analyze_fundamentals('INVALID')
        
        assert isinstance(result, dict)
        assert all(value == 0.0 for value in result.values())
    
    def test_analyze_market_sentiment(self, analyzer):
        """Test the analyze_market_sentiment method."""
        # Test with a valid ticker
        result = analyzer.analyze_market_sentiment('AAPL')
        
        assert isinstance(result, dict)
        assert 'analyst_rating' in result
        assert 'analyst_price_target' in result
        assert 'analyst_recommendation' in result
        assert 'institutional_ownership' in result
        assert 'insider_ownership' in result
        
        # Test with an invalid ticker
        result = analyzer.analyze_market_sentiment('INVALID')
        
        assert isinstance(result, dict)
        assert all(value == 0.0 for value in result.values())
    
    def test_analyze_news_sentiment(self, analyzer):
        """Test the analyze_news_sentiment method."""
        # Test with a valid ticker
        result = analyzer.analyze_news_sentiment('AAPL')
        
        assert isinstance(result, dict)
        assert 'sentiment_score' in result
        assert 'sentiment_magnitude' in result
        assert 'sentiment_keywords' in result
        
        # Test with an invalid ticker
        result = analyzer.analyze_news_sentiment('INVALID')
        
        assert isinstance(result, dict)
        assert result['sentiment_score'] == 0.0
        assert result['sentiment_magnitude'] == 0.0
        assert result['sentiment_keywords'] == []
    
    def test_calculate_confidence_score(self, analyzer, sample_opportunity):
        """Test the _calculate_confidence_score method."""
        # Test with valid opportunity
        result = analyzer._calculate_confidence_score(sample_opportunity)
        
        assert isinstance(result, float)
        assert 0 <= result <= 1
        
        # Test with invalid opportunity
        result = analyzer._calculate_confidence_score({})
        
        assert isinstance(result, float)
        assert result == 0.0
    
    def test_determine_action(self, analyzer, sample_opportunity):
        """Test the _determine_action method."""
        # Test with valid opportunity
        result = analyzer._determine_action(sample_opportunity)
        
        assert isinstance(result, str)
        assert result in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
        
        # Test with invalid opportunity
        result = analyzer._determine_action({})
        
        assert isinstance(result, str)
        assert result == 'HOLD'
    
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
        # Create a sample history DataFrame
        history = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        })
        
        # Call the method
        result = analyzer._calculate_volatility_metrics(history)
        
        # Check the result
        assert isinstance(result, dict)
        assert 'std_dev' in result
        assert result['std_dev'] is not None
        assert isinstance(result['std_dev'], float)
    
    def test_calculate_risk_score(self, analyzer, sample_potential_securities, mock_yf_data):
        """Test the _calculate_risk_score method."""
        with patch.object(analyzer, '_fetch_yf_data', return_value=mock_yf_data):
            # Call the method
            security = sample_potential_securities[0]
            result = analyzer._calculate_risk_score(security)
            
            # Check the result
            assert isinstance(result, int)
            assert 0 <= result <= 100
    
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
            assert result is True
            assert output_file.exists()
            
            # Check the content of the file
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert 'analysis_timestamp' in saved_data
            assert 'risk_scored_securities' in saved_data
            assert len(saved_data['risk_scored_securities']) == 2
            assert all('risk_score' in item for item in saved_data['risk_scored_securities'])
            
            # Check that the scores are sorted (ascending)
            scores = [item['risk_score'] for item in saved_data['risk_scored_securities']]
            assert scores == sorted(scores) 
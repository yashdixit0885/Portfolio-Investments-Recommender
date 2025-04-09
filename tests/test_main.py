import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.main import run_analysis_cycle
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.trade_analyst.trade_analyzer import TradeAnalyzer

class TestMainApplication:
    """Critical tests for the main application."""
    
    @pytest.fixture
    def sample_securities_data(self, tmp_path):
        """Create a sample securities data CSV file for testing."""
        csv_path = tmp_path / "securities_data.csv"
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'],
            'Name': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Meta', 'Tesla', 'NVIDIA', 'JPMorgan', 'Visa', 'Walmart'],
            'Last': [150.0, 280.0, 2500.0, 3300.0, 330.0, 900.0, 700.0, 140.0, 220.0, 140.0],
            'Industry': ['Technology', 'Technology', 'Technology', 'Consumer Cyclical', 'Technology', 
                        'Consumer Cyclical', 'Technology', 'Financial Services', 'Financial Services', 'Consumer Defensive'],
            'Market Cap': [2500000000000, 2000000000000, 1800000000000, 1700000000000, 900000000000, 
                          800000000000, 500000000000, 400000000000, 450000000000, 350000000000],
            'Volume': [1000000, 800000, 500000, 400000, 300000, 200000, 150000, 100000, 80000, 50000],
            '50D Avg Vol': [900000, 750000, 450000, 350000, 250000, 180000, 130000, 90000, 70000, 45000],
            '200D Avg Vol': [850000, 700000, 400000, 300000, 200000, 150000, 120000, 80000, 65000, 40000],
            '50D MA': [145.0, 275.0, 2450.0, 3200.0, 310.0, 850.0, 650.0, 135.0, 210.0, 135.0],
            '200D MA': [140.0, 270.0, 2400.0, 3100.0, 300.0, 800.0, 600.0, 130.0, 200.0, 130.0],
            'Beta': [1.2, 0.9, 1.1, 1.3, 1.4, 2.0, 1.8, 1.2, 1.0, 0.7],
            '14D Rel Str': [55, 60, 45, 65, 70, 80, 75, 50, 55, 40]
        })
        df.to_csv(csv_path, index=False)
        return csv_path
    
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
    
    def test_run_analysis_cycle(self, sample_securities_data, sample_potential_securities, 
                               sample_risk_scored_securities, mock_stock_data, tmp_path):
        """Test the run_analysis_cycle function."""
        # Create temporary files
        high_potential_file = tmp_path / "high_potential_securities.json"
        risk_scored_file = tmp_path / "risk_scored_securities.json"
        trade_recommendations_file = tmp_path / "Trade_Recommendations_latest.csv"
        
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Copy the securities data file to the data directory
        import shutil
        shutil.copy(sample_securities_data, data_dir / "securities_data.csv")
        
        # Mock the InvestmentAnalyzer
        with patch('src.investment_analyst.investment_analyzer.InvestmentAnalyzer') as mock_investment_analyzer:
            # Configure the mock
            mock_investment_instance = MagicMock()
            mock_investment_instance.run_analysis.return_value = True
            mock_investment_instance.input_file = str(data_dir / "securities_data.csv")
            mock_investment_instance.output_file = str(high_potential_file)
            mock_investment_analyzer.return_value = mock_investment_instance
            
            # Mock the ResearchAnalyzer
            with patch('src.research_analyst.research_analyzer.ResearchAnalyzer') as mock_research_analyzer:
                # Configure the mock
                mock_research_instance = MagicMock()
                mock_research_instance.run_risk_analysis.return_value = True
                mock_research_instance.input_file = str(high_potential_file)
                mock_research_instance.output_file = str(risk_scored_file)
                mock_research_analyzer.return_value = mock_research_instance
                
                # Mock the TradeAnalyzer
                with patch('src.trade_analyst.trade_analyzer.TradeAnalyzer') as mock_trade_analyzer:
                    # Configure the mock
                    mock_trade_instance = MagicMock()
                    mock_trade_instance.generate_trade_signals.return_value = True
                    mock_trade_instance.input_file = str(risk_scored_file)
                    mock_trade_instance.output_file = str(trade_recommendations_file)
                    mock_trade_analyzer.return_value = mock_trade_instance
                    
                    # Mock the file paths in the main module
                    with patch('src.main.os.path.join') as mock_join:
                        # Configure the mock to return the correct paths
                        def mock_join_impl(*args):
                            if args[0] == 'data':
                                if args[1] == 'securities_data.csv':
                                    return str(data_dir / "securities_data.csv")
                                elif args[1] == 'high_potential_securities.json':
                                    return str(high_potential_file)
                                elif args[1] == 'risk_scored_securities.json':
                                    return str(risk_scored_file)
                            elif args[0] == 'output':
                                if args[1] == 'Trade_Recommendations_latest.csv':
                                    return str(trade_recommendations_file)
                            return os.path.join(*args)
                        
                        mock_join.side_effect = mock_join_impl
                        
                        # Mock the os.makedirs function
                        with patch('src.main.os.makedirs'):
                            # Run the analysis cycle
                            run_analysis_cycle()
                            
                            # Check that the analyzers were called
                            mock_investment_instance.run_analysis.assert_called_once()
                            mock_research_instance.run_risk_analysis.assert_called_once()
                            mock_trade_instance.generate_trade_signals.assert_called_once() 
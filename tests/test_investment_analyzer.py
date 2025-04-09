import os
import pytest
import pandas as pd
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer

@pytest.mark.critical
class TestInvestmentAnalyzer:
    """Critical tests for the Investment Analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an instance of InvestmentAnalyzer."""
        return InvestmentAnalyzer()
    
    @pytest.fixture
    def sample_securities_data(self):
        """Create a sample securities data DataFrame."""
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Name': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc'],
            'Last': [150.0, 280.0, 2500.0],
            'Industry': ['Technology', 'Technology', 'Technology'],
            'Market Cap': ['2.5T', '2.0T', '1.8T'],
            'Volume': [1000000, 800000, 600000],
            '50D Avg Vol': [900000, 700000, 500000],
            '200D Avg Vol': [850000, 650000, 450000],
            '50D MA': [145.0, 275.0, 2450.0],
            '200D MA': [140.0, 270.0, 2400.0],
            'Beta': [1.2, 1.1, 1.0],
            '14D Rel Str': [1.05, 1.03, 1.01]
        })
    
    def test_clean_numeric(self, analyzer):
        """Test the _clean_numeric method."""
        # Test with numeric string
        assert analyzer._clean_numeric('1,234.56') == 1234.56
        
        # Test with percentage string
        assert analyzer._clean_numeric('12.34%') == 12.34
        
        # Test with empty value
        assert analyzer._clean_numeric('') == 0.0
        
        # Test with None
        assert analyzer._clean_numeric(None) == 0.0
    
    def test_calculate_price_movement_potential(self, analyzer, sample_securities_data):
        """Test the _calculate_price_movement_potential method."""
        row = sample_securities_data.iloc[0]
        score = analyzer._calculate_price_movement_potential(row)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_identify_potential_movers(self, analyzer, sample_securities_data, tmp_path):
        """Test the identify_potential_movers method."""
        # Create a temporary CSV file
        input_file = tmp_path / "securities_data.csv"
        sample_securities_data.to_csv(input_file, index=False)
        
        # Monkeypatch the input file path
        analyzer.input_file = str(input_file)
        
        # Run the method
        result = analyzer.identify_potential_movers()
        
        assert isinstance(result, list)
        assert len(result) == len(sample_securities_data)
        assert all(isinstance(item, dict) for item in result)
        assert all('movement_potential_score' in item for item in result)
        assert all(0 <= item['movement_potential_score'] <= 1 for item in result)
        assert result == sorted(result, key=lambda x: x['movement_potential_score'], reverse=True)
    
    def test_save_potential_securities(self, analyzer, tmp_path):
        """Test the save_potential_securities method."""
        # Create sample securities
        securities = [
            {
                'Symbol': 'AAPL',
                'Name': 'Apple Inc',
                'movement_potential_score': 0.8
            },
            {
                'Symbol': 'MSFT',
                'Name': 'Microsoft Corp',
                'movement_potential_score': 0.7
            }
        ]
        
        # Set the output file path
        output_file = tmp_path / "high_potential_securities.json"
        analyzer.output_file = str(output_file)
        
        # Run the method
        analyzer.save_potential_securities(securities)
        
        # Check if the file was created
        assert output_file.exists()
        
        # Read the file and check its contents
        with open(output_file, 'r') as f:
            data = pd.read_json(f)
        
        assert len(data) == len(securities)
        assert all(col in data.columns for col in ['Symbol', 'Name', 'movement_potential_score'])
        assert data['movement_potential_score'].max() == 0.8
        assert data['movement_potential_score'].min() == 0.7 
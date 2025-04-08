# Investment Analyzer

A comprehensive investment analysis system that combines fundamental, technical, and sentiment analysis to generate trade recommendations. The system analyzes market data, company fundamentals, news sentiment, and technical indicators to provide actionable trading insights.

## Features

- **Research Analyzer**: 
  - Analyzes company fundamentals (P/E ratio, revenue growth, profit margins)
  - Evaluates market sentiment from news and social media
  - Calculates risk metrics (beta, alpha, Sharpe ratio, drawdown)
  - Provides position sizing recommendations based on risk tolerance

- **Trade Analyzer**: 
  - Evaluates technical indicators (RSI, MACD, Moving Averages)
  - Identifies swing trading patterns
  - Analyzes volume and price trends
  - Generates short-term trading signals

- **Investment Analyzer**: 
  - Identifies investment opportunities from securities data
  - Combines research and trade analysis
  - Generates comprehensive recommendations
  - Provides rationale for each recommendation

- **Portfolio Manager**: 
  - Manages trade execution
  - Handles position sizing
  - Sends email notifications for trade recommendations
  - Tracks portfolio performance

- **Automated File Management**: 
  - Archives recommendations at 5 PM MT
  - Cleans up at 7 AM MT
  - Maintains historical data in organized folders

## Project Structure

```
Trader/
├── src/
│   ├── investment_analyst/  # Investment opportunity identification
│   │   ├── investment_analyzer.py
│   │   └── news_analyzer.py
│   ├── research_analyst/    # Fundamental and sentiment analysis
│   │   └── research_analyzer.py
│   ├── trade_analyst/       # Technical analysis and pattern recognition
│   │   └── trade_analyzer.py
│   ├── portfolio_manager/   # Trade execution and position sizing
│   │   └── portfolio_manager.py
│   └── utils/               # Common utilities and helper functions
│       └── common.py
├── config/                  # Configuration files
│   ├── news_analyzer_config.py
│   ├── research_analyzer_config.py
│   └── proxies.txt
├── tests/                   # Test scripts
│   ├── test_csv_export.py
│   ├── test_investment_analyzer.py
│   ├── test_research_analyzer.py
│   └── test_trade_analyzer.py
├── data/                    # Data storage (not tracked in git)
├── Output/                  # Generated recommendations (not tracked in git)
├── Archive/                 # Archived recommendations (not tracked in git)
└── logs/                    # Application logs (not tracked in git)
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yashdixit0885/Trader.git
   cd Trader
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration:
   # - API keys for financial data
   # - Email credentials for notifications
   # - Analysis parameters
   ```

## Usage

### End-to-End Flow

1. **Data Collection and Analysis**:
   ```bash
   # Run the investment analyzer to generate recommendations
   python src/investment_analyst/investment_analyzer.py
   ```
   This will:
   - Analyze securities data
   - Generate trade recommendations
   - Save results to Output/trade_recommendations_YYYYMMDD.csv

2. **Review Recommendations**:
   - Check the generated CSV file in the Output directory
   - Each recommendation includes:
     - Ticker and company name
     - Action (BUY/SELL/HOLD)
     - Score and position size
     - Current price and market cap
     - Price and volume changes
     - Detailed rationale

3. **Portfolio Management**:
   ```bash
   # Run the portfolio manager to execute trades
   python src/portfolio_manager/portfolio_manager.py
   ```
   This will:
   - Process recommendations
   - Send email notifications
   - Update position tracking

### Individual Components

You can also run individual components for specific analysis:

1. **Research Analysis**:
   ```bash
   python src/research_analyst/research_analyzer.py
   ```

2. **Technical Analysis**:
   ```bash
   python src/trade_analyst/trade_analyzer.py
   ```

3. **News Analysis**:
   ```bash
   python src/investment_analyst/news_analyzer.py
   ```

### Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/
```

## File Management

The system automatically manages files:
- New recommendations are saved to the Output directory
- Files are archived at 5 PM Mountain Time
- The Output directory is cleaned at 7 AM Mountain Time
- Archived files are stored in date-based folders in the Archive directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT 
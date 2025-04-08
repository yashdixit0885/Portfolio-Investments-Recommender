# Investment Analyzer

A comprehensive investment analysis system that combines fundamental, technical, and sentiment analysis to generate trade recommendations.

## Features

- **Research Analyzer**: Analyzes company fundamentals, market sentiment, and risk metrics
- **Trade Analyzer**: Evaluates technical indicators and swing trading patterns
- **Investment Analyzer**: Identifies investment opportunities from securities data
- **Portfolio Manager**: Manages trade execution and position sizing
- **Automated File Management**: Archives recommendations at 5 PM MT and cleans up at 7 AM MT

## Project Structure

```
Trader/
├── src/
│   ├── investment_analyst/  # Investment opportunity identification
│   ├── research_analyst/    # Fundamental and sentiment analysis
│   ├── trade_analyst/       # Technical analysis and pattern recognition
│   ├── portfolio_manager/   # Trade execution and position sizing
│   └── utils/               # Common utilities and helper functions
├── data/                    # Data storage (not tracked in git)
├── Output/                  # Generated recommendations (not tracked in git)
├── Archive/                 # Archived recommendations (not tracked in git)
├── logs/                    # Application logs (not tracked in git)
└── tests/                   # Test scripts
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Trader.git
   cd Trader
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (if needed):
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

1. Run the investment analyzer:
   ```
   python src/investment_analyst/investment_analyzer.py
   ```

2. Run the research analyzer:
   ```
   python src/research_analyst/research_analyzer.py
   ```

3. Run the trade analyzer:
   ```
   python src/trade_analyst/trade_analyzer.py
   ```

4. Run the portfolio manager:
   ```
   python src/portfolio_manager/portfolio_manager.py
   ```

## License

MIT 
import time
import logging
from typing import List, Dict
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class RateLimitTester:
    def __init__(self):
        self.logger = setup_logging()
        self.test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
        self.results = []

    def test_rate(self, calls_per_second: float, num_requests: int = 20) -> Dict:
        """
        Test a specific rate limit with Yahoo Finance API
        
        Args:
            calls_per_second: Number of calls to make per second
            num_requests: Total number of requests to make
            
        Returns:
            Dictionary containing test results
        """
        self.logger.info(f"\nTesting rate limit of {calls_per_second} calls/second")
        
        interval = 1.0 / calls_per_second
        successful_calls = 0
        failed_calls = 0
        rate_limit_errors = 0
        start_time = time.time()
        
        for i in range(num_requests):
            ticker = self.test_tickers[i % len(self.test_tickers)]
            try:
                # Make both info and history calls
                stock = yf.Ticker(ticker)
                info = stock.info
                history = stock.history(period='1d')
                
                if info and not history.empty:
                    successful_calls += 2  # Count both calls
                else:
                    failed_calls += 2
                
                self.logger.debug(f"Request {i+1}/{num_requests} successful for {ticker}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "too many requests" in error_msg:
                    rate_limit_errors += 1
                    self.logger.warning(f"Rate limit hit on request {i+1} for {ticker}")
                else:
                    self.logger.error(f"Error on request {i+1} for {ticker}: {str(e)}")
                failed_calls += 2
            
            # Wait for the next interval
            if i < num_requests - 1:
                time.sleep(interval)
        
        end_time = time.time()
        total_time = end_time - start_time
        actual_rate = successful_calls / total_time if total_time > 0 else 0
        
        result = {
            'calls_per_second': calls_per_second,
            'total_requests': num_requests * 2,  # Multiply by 2 for both info and history calls
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'rate_limit_errors': rate_limit_errors,
            'total_time': total_time,
            'actual_rate': actual_rate
        }
        
        self.results.append(result)
        self.logger.info(f"Test completed for {calls_per_second} calls/second:")
        self.logger.info(f"  Successful calls: {successful_calls}")
        self.logger.info(f"  Failed calls: {failed_calls}")
        self.logger.info(f"  Rate limit errors: {rate_limit_errors}")
        self.logger.info(f"  Actual rate achieved: {actual_rate:.2f} calls/second")
        
        return result

    def run_tests(self, rates: List[float] = [4.0, 6.0, 8.0, 10.0]):
        """Run tests for multiple rate limits"""
        self.logger.info("Starting rate limit tests...")
        
        for rate in rates:
            try:
                self.test_rate(rate)
                # Add a cooldown period between tests
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Test failed for rate {rate}: {str(e)}")
        
        # Save results to CSV
        df = pd.DataFrame(self.results)
        df.to_csv('rate_limit_test_results.csv', index=False)
        self.logger.info("Test results saved to rate_limit_test_results.csv")

if __name__ == "__main__":
    tester = RateLimitTester()
    tester.run_tests() 
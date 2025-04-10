import time
from datetime import datetime, timedelta
import logging

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_second=1.0, max_retries=3, retry_delay=5.0):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum number of calls per second
            max_retries: Maximum number of retries for failed calls
            retry_delay: Initial delay between retries in seconds
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_call_time = datetime.now() - timedelta(seconds=self.min_interval)
        self.logger = logging.getLogger('rate_limiter')
        
    def wait_if_needed(self):
        """Wait if needed to respect rate limits."""
        now = datetime.now()
        elapsed = (now - self.last_call_time).total_seconds()
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        self.last_call_time = datetime.now()
        
    def call_with_retry(self, func, *args, **kwargs):
        """
        Call a function with rate limiting and retry logic.
        
        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                self.wait_if_needed()
                return func(*args, **kwargs)
            except Exception as e:
                if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"Rate limit exceeded, retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                raise  # Re-raise if not a rate limit error or last attempt 
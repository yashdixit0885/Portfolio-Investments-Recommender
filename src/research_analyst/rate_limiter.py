import time
from typing import Dict, Any
import logging
from threading import Lock

class RateLimiter:
    """Rate limiter for API calls with retry mechanism"""
    
    def __init__(self, calls_per_second: float = 2.0, max_retries: int = 3, retry_delay: float = 5.0):
        self.min_interval = 1.0 / calls_per_second
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_call_time: Dict[str, float] = {}
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)

    def wait_if_needed(self, key: str = "default") -> None:
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()
            if key in self.last_call_time:
                elapsed = current_time - self.last_call_time[key]
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
            self.last_call_time[key] = time.time()

    def call_with_retry(self, func: callable, *args, **kwargs) -> Any:
        """Execute a function with retry mechanism"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.wait_if_needed()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    retry_delay = self.retry_delay * (attempt + 1)  # Exponential backoff
                    self.logger.info(f"Rate limit hit, waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # If it's not a rate limit error, wait a shorter time
                    time.sleep(1)
        
        raise last_exception  # Re-raise the last exception if all retries failed 
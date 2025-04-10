"""
Rate Limiter: Controls API call frequency to prevent rate limiting.

This module provides a rate limiter class that helps manage API call frequency
to prevent hitting rate limits when making external API calls.
"""

import time
import logging
from typing import Any, Callable, Optional

class RateLimiter:
    """Controls the rate of API calls to prevent rate limiting."""
    
    def __init__(self, calls_per_second: float = 2.0, max_retries: int = 3, retry_delay: float = 5.0):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_second: Maximum number of API calls per second
            max_retries: Maximum number of retries for failed calls
            retry_delay: Delay between retries in seconds
        """
        self.calls_per_second = calls_per_second
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_call_time = 0.0
        self.logger = logging.getLogger('rate_limiter')
        
    def call_with_retry(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting and retry logic.
        
        Args:
            func: The function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                current_time = time.time()
                time_since_last_call = current_time - self.last_call_time
                if time_since_last_call < 1.0 / self.calls_per_second:
                    sleep_time = (1.0 / self.calls_per_second) - time_since_last_call
                    time.sleep(sleep_time)
                
                # Make the API call
                result = func(*args, **kwargs)
                self.last_call_time = time.time()
                return result
                
            except Exception as e:
                if "Rate limit" in str(e) and attempt < self.max_retries - 1:
                    self.logger.warning(f"Rate limit exceeded, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                    continue
                raise  # Re-raise if not a rate limit error or last attempt 
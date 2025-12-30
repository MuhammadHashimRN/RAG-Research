"""
Rate limiter utility to prevent hitting API limits.
"""

import time
from collections import deque
from typing import Optional

class RateLimiter:
    """
    Simple rate limiter to control API call frequency.
    """
    
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()
    
    def wait_if_needed(self):
        """
        Wait if necessary to stay within rate limit.
        """
        current_time = time.time()
        
        # Remove old call times outside the window
        while self.call_times and current_time - self.call_times[0] > self.time_window:
            self.call_times.popleft()
        
        # If at limit, wait until oldest call expires
        if len(self.call_times) >= self.max_calls:
            wait_time = self.time_window - (current_time - self.call_times[0]) + 0.1
            if wait_time > 0:
                print(f"⏳ Rate limiter: Waiting {wait_time:.1f}s to stay within limit...")
                time.sleep(wait_time)
                # Clean up again after waiting
                current_time = time.time()
                while self.call_times and current_time - self.call_times[0] > self.time_window:
                    self.call_times.popleft()
        
        # Record this call
        self.call_times.append(time.time())
    
    def reset(self):
        """Reset call history."""
        self.call_times.clear()

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter(max_calls: int = 10, time_window: float = 60.0) -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls, time_window)
    return _rate_limiter

def set_rate_limit(max_calls: int, time_window: float = 60.0):
    """Set global rate limit."""
    global _rate_limiter
    _rate_limiter = RateLimiter(max_calls, time_window)


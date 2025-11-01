import logging
import os
import sys
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache
import threading

# Configuration
class LogConfig:
    """Centralized logging configuration"""
    LOG_DIR = 'logs'
    MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 3
    LOG_LEVEL = logging.INFO
    CONSOLE_LEVEL = logging.INFO
    FILE_LEVEL = logging.DEBUG
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    ENABLE_JSON_LOGS = False  # Set to True for structured JSON logging
    USE_TIMED_ROTATION = False  # Set to True for daily rotation instead of size-based

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
            
        return json.dumps(log_data)

class ContextFilter(logging.Filter):
    """Filter to add contextual information to logs"""
    
    def __init__(self):
        super().__init__()
        self.context = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add thread-safe context data
        record.thread_name = threading.current_thread().name
        if hasattr(self.context, 'user_id'):
            record.user_id = self.context.user_id
        return True
    
    def set_context(self, **kwargs):
        """Set context variables for this thread"""
        for key, value in kwargs.items():
            setattr(self.context, key, value)

class LoggerManager:
    """Singleton logger manager for efficient logger creation and management"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = LogConfig()
            self.context_filter = ContextFilter()
            self._setup_logging_directory()
            self._configure_root_logger()
            LoggerManager._initialized = True
    
    def _setup_logging_directory(self) -> Path:
        """Create logging directory if it doesn't exist"""
        root_dir = Path(__file__).resolve().parent.parent
        log_dir = root_dir / self.config.LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def _get_log_file_path(self) -> Path:
        """Generate log file path with timestamp"""
        log_dir = self._setup_logging_directory()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return log_dir / f"app_{timestamp}.log"
    
    def _create_file_handler(self) -> logging.Handler:
        """Create appropriate file handler based on configuration"""
        log_file = self._get_log_file_path()
        
        if self.config.USE_TIMED_ROTATION:
            # Rotate daily at midnight
            handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=self.config.BACKUP_COUNT,
                encoding='utf-8'
            )
        else:
            # Rotate based on file size
            handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.MAX_LOG_SIZE,
                backupCount=self.config.BACKUP_COUNT,
                encoding='utf-8'
            )
        
        handler.setLevel(self.config.FILE_LEVEL)
        return handler
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """Create console handler with colored output support"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.config.CONSOLE_LEVEL)
        return handler
    
    def _get_formatter(self) -> logging.Formatter:
        """Get appropriate formatter based on configuration"""
        if self.config.ENABLE_JSON_LOGS:
            return JsonFormatter()
        else:
            return logging.Formatter(
                fmt='[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
                datefmt=self.config.DATE_FORMAT
            )
    
    def _configure_root_logger(self):
        """Configure the root logger with handlers and filters"""
        root_logger = logging.getLogger()
        
        # Clear existing handlers to avoid duplicates
        if root_logger.handlers:
            root_logger.handlers.clear()
        
        root_logger.setLevel(self.config.LOG_LEVEL)
        
        # Create and configure handlers
        formatter = self._get_formatter()
        
        file_handler = self._create_file_handler()
        file_handler.setFormatter(formatter)
        file_handler.addFilter(self.context_filter)
        
        console_handler = self._create_console_handler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(self.context_filter)
        
        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False
    
    @lru_cache(maxsize=128)
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a cached logger instance"""
        logger = logging.getLogger(name)
        return logger
    
    def set_context(self, **kwargs):
        """Set context for current thread"""
        self.context_filter.set_context(**kwargs)
    
    def log_performance(self, func):
        """Decorator for logging function performance"""
        def wrapper(*args, **kwargs):
            logger = self.get_logger(func.__module__)
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"{func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func.__name__} failed after {duration:.4f}s: {str(e)}")
                raise
        return wrapper

# Singleton instance
_logger_manager = LoggerManager()


# Convenience functions
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If name is None, uses the caller's module name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        logging.Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        name = caller_frame.f_globals.get('__name__', 'root')
    
    return _logger_manager.get_logger(name)

def set_log_context(**kwargs):
    """Set contextual data for logs in current thread"""
    _logger_manager.set_context(**kwargs)

def log_performance(func):
    """Decorator to log function execution time"""
    return _logger_manager.log_performance(func)

# Example usage
if __name__ == "__main__":
    # Get logger for current module
    logger = get_logger(__name__)
    
    # Basic logging
    logger.info("Application started")
    logger.debug("Debug information")
    logger.warning("Warning message")
    
    # Context logging
    set_log_context(user_id='12345', request_id='req-abc')
    logger.info("Processing user request")
    
    # Performance logging decorator
    @log_performance
    def slow_function():
        import time
        time.sleep(0.1)
        return "Done"
    
    result = slow_function()
    
    # Error logging with exception
    try:
        1 / 0
    except Exception as e:
        logger.exception("An error occurred")
    
    logger.info("Application finished")
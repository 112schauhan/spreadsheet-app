"""
Application settings and configuration management.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Application info
    APP_NAME: str = Field(default="Collaborative Spreadsheet", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")

    # CORS configuration
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ],
        env="CORS_ORIGINS"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        env="CORS_ALLOW_METHODS"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_HEADERS"
    )

    # WebSocket configuration
    WS_PING_INTERVAL: int = Field(default=30, env="WS_PING_INTERVAL")
    WS_PING_TIMEOUT: int = Field(default=10, env="WS_PING_TIMEOUT")
    WS_CLOSE_TIMEOUT: int = Field(default=10, env="WS_CLOSE_TIMEOUT")
    WS_MAX_CONNECTIONS: int = Field(default=1000, env="WS_MAX_CONNECTIONS")

    # Grid configuration
    GRID_MAX_ROWS: int = Field(default=100, env="GRID_MAX_ROWS")
    GRID_MAX_COLUMNS: int = Field(default=26, env="GRID_MAX_COLUMNS")
    GRID_DEFAULT_CELL_VALUE: str = Field(
        default="", env="GRID_DEFAULT_CELL_VALUE")

    # Collaboration settings
    USER_SESSION_TIMEOUT: int = Field(
        default=1440, env="USER_SESSION_TIMEOUT")  # 24 hours in minutes
    USER_PRESENCE_TIMEOUT: int = Field(
        default=5, env="USER_PRESENCE_TIMEOUT")   # 5 minutes
    MAX_UNDO_HISTORY: int = Field(default=100, env="MAX_UNDO_HISTORY")
    CONFLICT_RESOLUTION_TIMEOUT: int = Field(
        default=30, env="CONFLICT_RESOLUTION_TIMEOUT")  # 30 seconds

    # Performance settings
    OPERATION_BATCH_SIZE: int = Field(default=50, env="OPERATION_BATCH_SIZE")
    OPERATION_TIMEOUT: int = Field(
        default=30, env="OPERATION_TIMEOUT")  # 30 seconds
    CLEANUP_INTERVAL: int = Field(
        default=300, env="CLEANUP_INTERVAL")   # 5 minutes

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(
        default=60, env="RATE_LIMIT_WINDOW")  # 1 minute

    # Formula engine settings
    FORMULA_MAX_ITERATIONS: int = Field(
        default=100, env="FORMULA_MAX_ITERATIONS")
    FORMULA_TIMEOUT: float = Field(
        default=5.0, env="FORMULA_TIMEOUT")  # 5 seconds
    FORMULA_MAX_CELL_REFERENCES: int = Field(
        default=1000, env="FORMULA_MAX_CELL_REFERENCES")

    # Data validation
    MAX_CELL_VALUE_LENGTH: int = Field(
        default=32767, env="MAX_CELL_VALUE_LENGTH")  # Excel limit
    MAX_FORMULA_LENGTH: int = Field(default=8192, env="MAX_FORMULA_LENGTH")

    # File handling
    MAX_CSV_SIZE: int = Field(default=10 * 1024 * 1024,
                              env="MAX_CSV_SIZE")  # 10MB
    TEMP_FILE_LIFETIME: int = Field(
        default=3600, env="TEMP_FILE_LIFETIME")  # 1 hour

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    LOG_ROTATION: bool = Field(default=True, env="LOG_ROTATION")
    LOG_MAX_SIZE: str = Field(default="10MB", env="LOG_MAX_SIZE")
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")

    # Security
    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 24 hours

    # Database (for future use)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")

    # Redis (for future use)
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")

    # Monitoring and metrics
    ENABLE_METRICS: bool = Field(default=False, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")

    # Development settings
    ENABLE_DOCS: bool = Field(default=True, env="ENABLE_DOCS")
    ENABLE_PROFILER: bool = Field(default=False, env="ENABLE_PROFILER")

    @validator('CORS_ORIGINS', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v

    @validator('CORS_ALLOW_METHODS', pre=True)
    def parse_cors_methods(cls, v):
        """Parse CORS methods from string or list."""
        if isinstance(v, str):
            return [method.strip().upper() for method in v.split(',') if method.strip()]
        return v

    @validator('CORS_ALLOW_HEADERS', pre=True)
    def parse_cors_headers(cls, v):
        """Parse CORS headers from string or list."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(',') if header.strip()]
        return v

    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    @validator('GRID_MAX_ROWS')
    def validate_max_rows(cls, v):
        """Validate maximum rows."""
        if v < 1 or v > 1000:
            raise ValueError('Maximum rows must be between 1 and 1000')
        return v

    @validator('GRID_MAX_COLUMNS')
    def validate_max_columns(cls, v):
        """Validate maximum columns."""
        if v < 1 or v > 26:
            raise ValueError('Maximum columns must be between 1 and 26 (A-Z)')
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.DEBUG

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG

    def get_cors_config(self) -> dict:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.CORS_ORIGINS,
            "allow_credentials": self.CORS_ALLOW_CREDENTIALS,
            "allow_methods": self.CORS_ALLOW_METHODS,
            "allow_headers": self.CORS_ALLOW_HEADERS,
        }

    def get_websocket_config(self) -> dict:
        """Get WebSocket configuration dictionary."""
        return {
            "ping_interval": self.WS_PING_INTERVAL,
            "ping_timeout": self.WS_PING_TIMEOUT,
            "close_timeout": self.WS_CLOSE_TIMEOUT,
            "max_connections": self.WS_MAX_CONNECTIONS,
        }

    def get_grid_config(self) -> dict:
        """Get grid configuration dictionary."""
        return {
            "max_rows": self.GRID_MAX_ROWS,
            "max_columns": self.GRID_MAX_COLUMNS,
            "default_cell_value": self.GRID_DEFAULT_CELL_VALUE,
        }

    def get_collaboration_config(self) -> dict:
        """Get collaboration configuration dictionary."""
        return {
            "session_timeout": self.USER_SESSION_TIMEOUT,
            "presence_timeout": self.USER_PRESENCE_TIMEOUT,
            "max_undo_history": self.MAX_UNDO_HISTORY,
            "conflict_resolution_timeout": self.CONFLICT_RESOLUTION_TIMEOUT,
        }

    def get_performance_config(self) -> dict:
        """Get performance configuration dictionary."""
        return {
            "operation_batch_size": self.OPERATION_BATCH_SIZE,
            "operation_timeout": self.OPERATION_TIMEOUT,
            "cleanup_interval": self.CLEANUP_INTERVAL,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

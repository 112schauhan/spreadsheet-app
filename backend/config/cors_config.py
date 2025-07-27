"""
CORS (Cross-Origin Resource Sharing) configuration.
"""

from typing import List, Dict, Any, Union
from .settings import get_settings


def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration for FastAPI."""
    settings = get_settings()
    
    return {
        "allow_origins": settings.CORS_ORIGINS,
        "allow_credentials": settings.CORS_ALLOW_CREDENTIALS,
        "allow_methods": settings.CORS_ALLOW_METHODS,
        "allow_headers": settings.CORS_ALLOW_HEADERS,
        "expose_headers": [
            "Content-Length",
            "Content-Type",
            "X-Total-Count",
            "X-Request-ID"
        ],
        "max_age": 3600  # Cache preflight response for 1 hour
    }


def get_development_cors_config() -> Dict[str, Any]:
    """Get permissive CORS configuration for development."""
    return {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": ["*"],
        "max_age": 86400  # Cache for 24 hours in development
    }


def get_production_cors_config(allowed_origins: List[str]) -> Dict[str, Any]:
    """Get restrictive CORS configuration for production."""
    return {
        "allow_origins": allowed_origins,
        "allow_credentials": True,
        "allow_methods": [
            "GET",
            "POST", 
            "PUT",
            "DELETE",
            "OPTIONS",
            "PATCH"
        ],
        "allow_headers": [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "Cache-Control"
        ],
        "expose_headers": [
            "Content-Length",
            "Content-Type",
            "X-Total-Count",
            "X-Request-ID"
        ],
        "max_age": 3600
    }


def validate_cors_origins(origins: List[str]) -> List[str]:
    """Validate and normalize CORS origins."""
    validated_origins = []
    
    for origin in origins:
        origin = origin.strip()
        if not origin:
            continue
            
        # Allow wildcard in development
        if origin == "*":
            validated_origins.append(origin)
            continue
        
        # Validate URL format
        if not (origin.startswith("http://") or origin.startswith("https://")):
            # Assume http for localhost
            if "localhost" in origin or "127.0.0.1" in origin:
                origin = f"http://{origin}"
            else:
                origin = f"https://{origin}"
        
        # Remove trailing slash
        origin = origin.rstrip("/")
        validated_origins.append(origin)
    
    return validated_origins


class CORSConfigBuilder:
    """Builder class for creating CORS configurations."""
    
    def __init__(self):
        self.config = {
            "allow_origins": [],
            "allow_credentials": False,
            "allow_methods": ["GET"],
            "allow_headers": ["Content-Type"],
            "expose_headers": [],
            "max_age": 600
        }
    
    def allow_origins(self, origins: Union[str, List[str]]) -> 'CORSConfigBuilder':
        """Set allowed origins."""
        if isinstance(origins, str):
            origins = [origins]
        
        self.config["allow_origins"] = validate_cors_origins(origins)
        return self
    
    def allow_credentials(self, allow: bool = True) -> 'CORSConfigBuilder':
        """Set whether to allow credentials."""
        self.config["allow_credentials"] = allow
        return self
    
    def allow_methods(self, methods: Union[str, List[str]]) -> 'CORSConfigBuilder':
        """Set allowed HTTP methods."""
        if isinstance(methods, str):
            methods = [methods]
        
        self.config["allow_methods"] = [method.upper() for method in methods]
        return self
    
    def allow_headers(self, headers: Union[str, List[str]]) -> 'CORSConfigBuilder':
        """Set allowed headers."""
        if isinstance(headers, str):
            headers = [headers]
        
        self.config["allow_headers"] = headers
        return self
    
    def expose_headers(self, headers: Union[str, List[str]]) -> 'CORSConfigBuilder':
        """Set exposed headers."""
        if isinstance(headers, str):
            headers = [headers]
        
        self.config["expose_headers"] = headers
        return self
    
    def max_age(self, age: int) -> 'CORSConfigBuilder':
        """Set max age for preflight cache."""
        self.config["max_age"] = age
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the CORS configuration."""
        return self.config.copy()
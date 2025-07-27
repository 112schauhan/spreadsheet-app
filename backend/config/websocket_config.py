"""
WebSocket configuration for real-time collaboration.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

from .settings import get_settings


class WebSocketEventType(str, Enum):
    """WebSocket event types for real-time collaboration."""
    
    # Connection events
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"
    
    # Authentication events
    AUTHENTICATE = "authenticate"
    AUTHENTICATED = "authenticated"
    AUTH_ERROR = "auth_error"
    
    # Cell events
    CELL_UPDATE = "cell_update"
    CELL_SELECT = "cell_select"
    CELL_EDIT_START = "cell_edit_start"
    CELL_EDIT_END = "cell_edit_end"
    CELL_FORMAT = "cell_format"
    
    # Range events
    RANGE_SELECT = "range_select"
    RANGE_UPDATE = "range_update"
    RANGE_COPY = "range_copy"
    RANGE_PASTE = "range_paste"
    
    # Grid operation events
    ROW_INSERT = "row_insert"
    ROW_DELETE = "row_delete"
    COLUMN_INSERT = "column_insert"
    COLUMN_DELETE = "column_delete"
    SORT_COLUMN = "sort_column"
    
    # Formula events
    FORMULA_UPDATE = "formula_update"
    FORMULA_CALCULATE = "formula_calculate"
    FORMULA_ERROR = "formula_error"
    
    # Collaboration events
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    USER_LIST = "user_list"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    
    # Conflict resolution events
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    
    # System events
    GRID_STATE = "grid_state"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    
    # Undo/Redo events
    UNDO = "undo"
    REDO = "redo"
    HISTORY_UPDATE = "history_update"


class WebSocketMessage(BaseModel):
    """Standard WebSocket message format."""
    
    event_type: WebSocketEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None  # For request-response correlation


class WebSocketConfig(BaseModel):
    """WebSocket configuration settings."""
    
    # Connection settings
    ping_interval: int = Field(default=30, description="Ping interval in seconds")
    ping_timeout: int = Field(default=10, description="Ping timeout in seconds")
    close_timeout: int = Field(default=10, description="Close timeout in seconds")
    max_connections: int = Field(default=1000, description="Maximum concurrent connections")
    
    # Message settings
    max_message_size: int = Field(default=64 * 1024, description="Maximum message size in bytes")
    message_queue_size: int = Field(default=100, description="Maximum queued messages per connection")
    
    # Heartbeat settings
    heartbeat_interval: int = Field(default=30, description="Heartbeat interval in seconds")
    connection_timeout: int = Field(default=300, description="Connection timeout in seconds")
    
    # Rate limiting
    rate_limit_messages: int = Field(default=100, description="Messages per minute per connection")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Collaboration settings
    broadcast_debounce: float = Field(default=0.1, description="Broadcast debounce in seconds")
    conflict_resolution_timeout: int = Field(default=30, description="Conflict resolution timeout")
    
    # Authentication
    require_auth: bool = Field(default=False, description="Require authentication")
    auth_timeout: int = Field(default=30, description="Authentication timeout in seconds")
    
    @classmethod
    def from_settings(cls) -> 'WebSocketConfig':
        """Create WebSocket config from application settings."""
        settings = get_settings()
        return cls(
            ping_interval=settings.WS_PING_INTERVAL,
            ping_timeout=settings.WS_PING_TIMEOUT,
            close_timeout=settings.WS_CLOSE_TIMEOUT,
            max_connections=settings.WS_MAX_CONNECTIONS,
            conflict_resolution_timeout=settings.CONFLICT_RESOLUTION_TIMEOUT,
        )


class WebSocketErrorCode(str, Enum):
    """WebSocket error codes."""
    
    # Connection errors
    CONNECTION_FAILED = "connection_failed"
    CONNECTION_TIMEOUT = "connection_timeout"
    CONNECTION_LIMIT_EXCEEDED = "connection_limit_exceeded"
    
    # Authentication errors
    AUTH_REQUIRED = "auth_required"
    AUTH_FAILED = "auth_failed"
    AUTH_EXPIRED = "auth_expired"
    INVALID_TOKEN = "invalid_token"
    
    # Message errors
    INVALID_MESSAGE_FORMAT = "invalid_message_format"
    MESSAGE_TOO_LARGE = "message_too_large"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Operation errors
    INVALID_OPERATION = "invalid_operation"
    OPERATION_FAILED = "operation_failed"
    PERMISSION_DENIED = "permission_denied"
    
    # Grid errors
    INVALID_CELL_ADDRESS = "invalid_cell_address"
    INVALID_RANGE = "invalid_range"
    CELL_LOCKED = "cell_locked"
    FORMULA_ERROR = "formula_error"
    
    # Collaboration errors
    CONFLICT_DETECTED = "conflict_detected"
    USER_NOT_FOUND = "user_not_found"
    SESSION_EXPIRED = "session_expired"
    
    # System errors
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"


class WebSocketResponse:
    """Helper class for creating WebSocket responses."""
    
    @staticmethod
    def success(event_type: WebSocketEventType, data: Dict[str, Any] = None, 
               user_id: str = None, correlation_id: str = None) -> Dict[str, Any]:
        """Create a success response."""
        message = {
            "event_type": event_type.value,
            "data": data or {},
            "success": True
        }
        
        if user_id:
            message["user_id"] = user_id
        if correlation_id:
            message["correlation_id"] = correlation_id
            
        return message
    
    @staticmethod
    def error(error_code: WebSocketErrorCode, message: str = None, 
             data: Dict[str, Any] = None, correlation_id: str = None) -> Dict[str, Any]:
        """Create an error response."""
        response = {
            "event_type": WebSocketEventType.ERROR.value,
            "success": False,
            "error": {
                "code": error_code.value,
                "message": message or error_code.value.replace("_", " ").title()
            }
        }
        
        if data:
            response["data"] = data
        if correlation_id:
            response["correlation_id"] = correlation_id
            
        return response
    
    @staticmethod
    def cell_update(cell_address: str, value: Any, user_id: str, 
                   formula: str = None, formatting: Dict = None) -> Dict[str, Any]:
        """Create a cell update response."""
        return WebSocketResponse.success(
            WebSocketEventType.CELL_UPDATE,
            {
                "address": cell_address,
                "value": value,
                "formula": formula,
                "formatting": formatting,
                "timestamp": None  # Will be set by handler
            },
            user_id
        )
    
    @staticmethod
    def user_join(user_id: str, username: str, color: str, 
                 cursor_position: Dict = None) -> Dict[str, Any]:
        """Create a user join response."""
        return WebSocketResponse.success(
            WebSocketEventType.USER_JOIN,
            {
                "user_id": user_id,
                "username": username,
                "color": color,
                "cursor_position": cursor_position
            }
        )
    
    @staticmethod
    def user_leave(user_id: str) -> Dict[str, Any]:
        """Create a user leave response."""
        return WebSocketResponse.success(
            WebSocketEventType.USER_LEAVE,
            {"user_id": user_id}
        )
    
    @staticmethod
    def cursor_move(user_id: str, row: int, column: int) -> Dict[str, Any]:
        """Create a cursor move response."""
        return WebSocketResponse.success(
            WebSocketEventType.CURSOR_MOVE,
            {
                "user_id": user_id,
                "position": {"row": row, "column": column}
            }
        )
    
    @staticmethod
    def selection_change(user_id: str, start_address: str, 
                        end_address: str = None) -> Dict[str, Any]:
        """Create a selection change response."""
        return WebSocketResponse.success(
            WebSocketEventType.SELECTION_CHANGE,
            {
                "user_id": user_id,
                "selection": {
                    "start": start_address,
                    "end": end_address or start_address
                }
            }
        )
    
    @staticmethod
    def grid_state(grid_data: Dict[str, Any], active_users: List[Dict]) -> Dict[str, Any]:
        """Create a grid state response."""
        return WebSocketResponse.success(
            WebSocketEventType.GRID_STATE,
            {
                "grid": grid_data,
                "users": active_users
            }
        )
    
    @staticmethod
    def conflict_detected(operation_id: str, conflicting_user: str, 
                         cell_address: str) -> Dict[str, Any]:
        """Create a conflict detected response."""
        return WebSocketResponse.success(
            WebSocketEventType.CONFLICT_DETECTED,
            {
                "operation_id": operation_id,
                "conflicting_user": conflicting_user,
                "cell_address": cell_address
            }
        )


class WebSocketConnectionManager:
    """Configuration for WebSocket connection management."""
    
    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig.from_settings()
        self.active_connections: Dict[str, Any] = {}
        self.user_connections: Dict[str, List[str]] = {}
        self.connection_count = 0
    
    def can_accept_connection(self) -> bool:
        """Check if server can accept new connections."""
        return self.connection_count < self.config.max_connections
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information."""
        return {
            "active_connections": self.connection_count,
            "max_connections": self.config.max_connections,
            "unique_users": len(self.user_connections),
            "config": self.config.dict()
        }


def get_websocket_config() -> WebSocketConfig:
    """Get WebSocket configuration instance."""
    return WebSocketConfig.from_settings()
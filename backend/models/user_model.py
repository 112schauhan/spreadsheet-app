"""
User and session models for collaborative spreadsheet application.
Handles user presence, sessions, and collaboration state.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import uuid
import random


class UserStatus(str, Enum):
    """User presence status."""
    ONLINE = "online"
    AWAY = "away"
    OFFLINE = "offline"


class UserRole(str, Enum):
    """User role in collaboration."""
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


class CursorPosition(BaseModel):
    """User's cursor position in the grid."""
    row: int = Field(ge=0, le=99)
    column: int = Field(ge=0, le=25)
    
    @property
    def address(self) -> str:
        """Get cursor position as cell address."""
        column_letter = chr(ord('A') + self.column)
        return f"{column_letter}{self.row + 1}"
    
    @classmethod
    def from_address(cls, address: str) -> 'CursorPosition':
        """Create cursor position from cell address."""
        from .cell_model import Cell
        row, column = Cell.parse_address(address)
        return cls(row=row, column=column)


class UserPresence(BaseModel):
    """User presence information for real-time collaboration."""
    
    user_id: str
    status: UserStatus = UserStatus.ONLINE
    cursor_position: Optional[CursorPosition] = None
    selected_range: Optional[Dict[str, str]] = None  # {"start": "A1", "end": "B2"}
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    color: str = Field(default_factory=lambda: UserPresence._generate_user_color())
    
    @staticmethod
    def _generate_user_color() -> str:
        """Generate a random color for user identification."""
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
            "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43",
            "#F368E0", "#576574", "#3742FA", "#2F3542", "#FF3838"
        ]
        return random.choice(colors)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        if self.status == UserStatus.OFFLINE:
            self.status = UserStatus.ONLINE
    
    def is_active(self, timeout_minutes: int = 5) -> bool:
        """Check if user is considered active."""
        if self.status == UserStatus.OFFLINE:
            return False
        
        timeout = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        return self.last_activity > timeout


class UserSession(BaseModel):
    """User session information."""
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    websocket_id: Optional[str] = None
    
    # Session metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Connection info
    is_connected: bool = True
    connection_count: int = 1
    
    def update_activity(self) -> None:
        """Update session activity."""
        self.last_activity = datetime.utcnow()
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session is expired."""
        timeout = datetime.utcnow() - timedelta(hours=timeout_hours)
        return self.last_activity < timeout


class User(BaseModel):
    """
    User model for collaborative spreadsheet application.
    Represents a user participating in real-time collaboration.
    """
    
    # Basic user info
    user_id: str = Field(default_factory=lambda: f"user_{uuid.uuid4().hex[:8]}")
    username: str = Field(default="Anonymous User")
    email: Optional[str] = None
    
    # Collaboration settings
    role: UserRole = UserRole.EDITOR
    color: str = Field(default_factory=lambda: UserPresence._generate_user_color())
    
    # Presence and session
    presence: UserPresence = Field(default=None)
    current_session: Optional[UserSession] = None
    sessions: List[UserSession] = Field(default_factory=list)
    
    # Permissions and preferences
    can_edit: bool = True
    can_view_formulas: bool = True
    can_add_comments: bool = True
    
    # Statistics
    total_edits: int = 0
    last_edit: Optional[datetime] = None
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('presence', pre=True, always=True)
    def create_presence(cls, v, values):
        """Create user presence if not provided."""
        if v is None and 'user_id' in values:
            return UserPresence(user_id=values['user_id'])
        return v
    
    def create_session(self, websocket_id: str = None, user_agent: str = None, 
                      ip_address: str = None) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=self.user_id,
            websocket_id=websocket_id,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        self.sessions.append(session)
        self.current_session = session
        
        # Update presence
        if self.presence:
            self.presence.status = UserStatus.ONLINE
            self.presence.update_activity()
        
        return session
    
    def end_session(self, session_id: str = None) -> bool:
        """End a user session."""
        if session_id is None and self.current_session:
            session_id = self.current_session.session_id
        
        for session in self.sessions:
            if session.session_id == session_id:
                session.is_connected = False
                if self.current_session and self.current_session.session_id == session_id:
                    self.current_session = None
                
                # Update presence if no active sessions
                if not self.has_active_sessions():
                    if self.presence:
                        self.presence.status = UserStatus.OFFLINE
                
                return True
        
        return False
    
    def has_active_sessions(self) -> bool:
        """Check if user has any active sessions."""
        return any(session.is_connected and not session.is_expired() 
                  for session in self.sessions)
    
    def update_cursor_position(self, row: int, column: int) -> None:
        """Update user's cursor position."""
        if self.presence:
            self.presence.cursor_position = CursorPosition(row=row, column=column)
            self.presence.update_activity()
    
    def update_selection(self, start_address: str, end_address: str = None) -> None:
        """Update user's selected range."""
        if self.presence:
            if end_address is None:
                end_address = start_address
            
            self.presence.selected_range = {
                "start": start_address,
                "end": end_address
            }
            self.presence.update_activity()
    
    def clear_selection(self) -> None:
        """Clear user's current selection."""
        if self.presence:
            self.presence.selected_range = None
            self.presence.update_activity()
    
    def record_edit(self) -> None:
        """Record that user made an edit."""
        self.total_edits += 1
        self.last_edit = datetime.utcnow()
        
        if self.presence:
            self.presence.update_activity()
    
    def set_status(self, status: UserStatus) -> None:
        """Set user status."""
        if self.presence:
            self.presence.status = status
            if status == UserStatus.ONLINE:
                self.presence.update_activity()
    
    def get_active_session(self) -> Optional[UserSession]:
        """Get the current active session."""
        if self.current_session and self.current_session.is_connected:
            return self.current_session
        return None
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count removed."""
        initial_count = len(self.sessions)
        self.sessions = [s for s in self.sessions if not s.is_expired()]
        
        # Clear current session if expired
        if (self.current_session and 
            self.current_session.is_expired()):
            self.current_session = None
        
        return initial_count - len(self.sessions)
    
    def to_dict(self, include_sessions: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary for JSON serialization."""
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "color": self.color,
            "can_edit": self.can_edit,
            "can_view_formulas": self.can_view_formulas,
            "can_add_comments": self.can_add_comments,
            "stats": {
                "total_edits": self.total_edits,
                "last_edit": self.last_edit.isoformat() if self.last_edit else None,
                "joined_at": self.joined_at.isoformat()
            }
        }
        
        if self.presence:
            data["presence"] = {
                "status": self.presence.status.value,
                "cursor_position": (
                    self.presence.cursor_position.dict() 
                    if self.presence.cursor_position else None
                ),
                "selected_range": self.presence.selected_range,
                "last_activity": self.presence.last_activity.isoformat(),
                "color": self.presence.color
            }
        
        if include_sessions:
            data["sessions"] = [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at.isoformat(),
                    "last_activity": s.last_activity.isoformat(),
                    "is_connected": s.is_connected,
                    "user_agent": s.user_agent,
                    "ip_address": s.ip_address
                }
                for s in self.sessions
            ]
        
        return data
    
    @classmethod
    def create_anonymous(cls, username: str = None) -> 'User':
        """Create an anonymous user."""
        user_id = f"anon_{uuid.uuid4().hex[:8]}"
        username = username or f"Anonymous User {user_id[-4:]}"
        
        return cls(
            user_id=user_id,
            username=username,
            role=UserRole.EDITOR
        )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
"""
Cursor position and selection models for real-time collaboration.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class CursorState(str, Enum):
    """States of user cursor."""
    IDLE = "idle"
    SELECTING = "selecting"
    EDITING = "editing"
    MOVING = "moving"
    COPYING = "copying"
    PASTING = "pasting"


class SelectionType(str, Enum):
    """Types of selections."""
    SINGLE_CELL = "single_cell"
    RANGE = "range"
    ROW = "row"
    COLUMN = "column"
    MULTIPLE = "multiple"


class CursorPosition(BaseModel):
    """Represents a cursor position in the grid."""

    row: int = Field(ge=0, le=99, description="Row index (0-99)")
    column: int = Field(ge=0, le=25, description="Column index (0-25)")

    @property
    def address(self) -> str:
        """Get cursor position as cell address (e.g., A1, B2)."""
        column_letter = chr(ord('A') + self.column)
        return f"{column_letter}{self.row + 1}"

    @property
    def coordinates(self) -> Tuple[int, int]:
        """Get cursor position as (row, column) tuple."""
        return (self.row, self.column)

    @classmethod
    def from_address(cls, address: str) -> 'CursorPosition':
        """Create cursor position from cell address."""
        from models.cell_model import Cell
        row, column = Cell.parse_address(address)
        return cls(row=row, column=column)

    @classmethod
    def from_coordinates(cls, row: int, column: int) -> 'CursorPosition':
        """Create cursor position from row and column."""
        return cls(row=row, column=column)

    def move_to(self, row: int, column: int) -> 'CursorPosition':
        """Move cursor to new position."""
        return CursorPosition(row=row, column=column)

    def move_by(self, row_delta: int, column_delta: int) -> 'CursorPosition':
        """Move cursor by delta values."""
        new_row = max(0, min(99, self.row + row_delta))
        new_column = max(0, min(25, self.column + column_delta))
        return CursorPosition(row=new_row, column=new_column)

    def move_up(self, steps: int = 1) -> 'CursorPosition':
        """Move cursor up."""
        return self.move_by(-steps, 0)

    def move_down(self, steps: int = 1) -> 'CursorPosition':
        """Move cursor down."""
        return self.move_by(steps, 0)

    def move_left(self, steps: int = 1) -> 'CursorPosition':
        """Move cursor left."""
        return self.move_by(0, -steps)

    def move_right(self, steps: int = 1) -> 'CursorPosition':
        """Move cursor right."""
        return self.move_by(0, steps)

    def is_valid(self) -> bool:
        """Check if cursor position is valid."""
        return 0 <= self.row <= 99 and 0 <= self.column <= 25

    def distance_to(self, other: 'CursorPosition') -> float:
        """Calculate distance to another cursor position."""
        import math
        return math.sqrt((self.row - other.row)**2 + (self.column - other.column)**2)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CursorPosition):
            return False
        return self.row == other.row and self.column == other.column

    def __hash__(self) -> int:
        return hash((self.row, self.column))

    def __str__(self) -> str:
        return self.address

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "row": self.row,
            "column": self.column,
            "address": self.address,
            "coordinates": self.coordinates
        }


class CellRange(BaseModel):
    """Represents a range of cells."""

    start_position: CursorPosition
    end_position: CursorPosition

    @property
    def start_address(self) -> str:
        """Get start address of range."""
        return self.start_position.address

    @property
    def end_address(self) -> str:
        """Get end address of range."""
        return self.end_position.address

    @property
    def range_notation(self) -> str:
        """Get range in A1:B2 notation."""
        return f"{self.start_address}:{self.end_address}"

    @property
    def normalized_range(self) -> 'CellRange':
        """Get normalized range with start <= end."""
        min_row = min(self.start_position.row, self.end_position.row)
        max_row = max(self.start_position.row, self.end_position.row)
        min_col = min(self.start_position.column, self.end_position.column)
        max_col = max(self.start_position.column, self.end_position.column)

        return CellRange(
            start_position=CursorPosition(row=min_row, column=min_col),
            end_position=CursorPosition(row=max_row, column=max_col)
        )

    @property
    def cell_count(self) -> int:
        """Get number of cells in range."""
        normalized = self.normalized_range
        rows = normalized.end_position.row - normalized.start_position.row + 1
        cols = normalized.end_position.column - normalized.start_position.column + 1
        return rows * cols

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get range dimensions as (rows, columns)."""
        normalized = self.normalized_range
        rows = normalized.end_position.row - normalized.start_position.row + 1
        cols = normalized.end_position.column - normalized.start_position.column + 1
        return (rows, cols)

    @classmethod
    def from_addresses(cls, start_address: str, end_address: str) -> 'CellRange':
        """Create range from cell addresses."""
        return cls(
            start_position=CursorPosition.from_address(start_address),
            end_position=CursorPosition.from_address(end_address)
        )

    @classmethod
    def from_single_cell(cls, address: str) -> 'CellRange':
        """Create single-cell range."""
        position = CursorPosition.from_address(address)
        return cls(start_position=position, end_position=position)

    @classmethod
    def from_coordinates(cls, start_row: int, start_col: int,
                         end_row: int, end_col: int) -> 'CellRange':
        """Create range from coordinates."""
        return cls(
            start_position=CursorPosition(row=start_row, column=start_col),
            end_position=CursorPosition(row=end_row, column=end_col)
        )

    def contains_position(self, position: CursorPosition) -> bool:
        """Check if range contains a position."""
        normalized = self.normalized_range
        return (
            normalized.start_position.row <= position.row <= normalized.end_position.row and
            normalized.start_position.column <= position.column <= normalized.end_position.column
        )

    def contains_address(self, address: str) -> bool:
        """Check if range contains a cell address."""
        position = CursorPosition.from_address(address)
        return self.contains_position(position)

    def intersects_with(self, other: 'CellRange') -> bool:
        """Check if this range intersects with another range."""
        self_norm = self.normalized_range
        other_norm = other.normalized_range

        return not (
            self_norm.end_position.row < other_norm.start_position.row or
            self_norm.start_position.row > other_norm.end_position.row or
            self_norm.end_position.column < other_norm.start_position.column or
            self_norm.start_position.column > other_norm.end_position.column
        )

    def get_all_positions(self) -> List[CursorPosition]:
        """Get all positions in the range."""
        normalized = self.normalized_range
        positions = []

        for row in range(normalized.start_position.row, normalized.end_position.row + 1):
            for col in range(normalized.start_position.column, normalized.end_position.column + 1):
                positions.append(CursorPosition(row=row, column=col))

        return positions

    def get_all_addresses(self) -> List[str]:
        """Get all cell addresses in the range."""
        return [pos.address for pos in self.get_all_positions()]

    def expand_to_include(self, position: CursorPosition) -> 'CellRange':
        """Expand range to include a position."""
        min_row = min(self.start_position.row,
                      self.end_position.row, position.row)
        max_row = max(self.start_position.row,
                      self.end_position.row, position.row)
        min_col = min(self.start_position.column,
                      self.end_position.column, position.column)
        max_col = max(self.start_position.column,
                      self.end_position.column, position.column)

        return CellRange(
            start_position=CursorPosition(row=min_row, column=min_col),
            end_position=CursorPosition(row=max_row, column=max_col)
        )

    def is_single_cell(self) -> bool:
        """Check if range is a single cell."""
        return self.start_position == self.end_position

    def is_single_row(self) -> bool:
        """Check if range is a single row."""
        return self.start_position.row == self.end_position.row

    def is_single_column(self) -> bool:
        """Check if range is a single column."""
        return self.start_position.column == self.end_position.column

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_address": self.start_address,
            "end_address": self.end_address,
            "range_notation": self.range_notation,
            "start_position": self.start_position.to_dict(),
            "end_position": self.end_position.to_dict(),
            "cell_count": self.cell_count,
            "dimensions": self.dimensions,
            "is_single_cell": self.is_single_cell(),
            "is_single_row": self.is_single_row(),
            "is_single_column": self.is_single_column()
        }


class UserCursor(BaseModel):
    """Represents a user's cursor state and selection."""

    # User identification
    user_id: str
    user_name: str = Field(default="Unknown User")
    user_color: str = Field(default="#000000")

    # Cursor state
    position: CursorPosition
    state: CursorState = CursorState.IDLE

    # Selection
    selection: Optional[CellRange] = None
    selection_type: SelectionType = SelectionType.SINGLE_CELL

    # Timing
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    last_moved: datetime = Field(default_factory=datetime.utcnow)

    # Visibility
    is_visible: bool = True
    is_active: bool = True

    # Edit state
    is_editing: bool = False
    editing_cell: Optional[str] = None
    edit_start_time: Optional[datetime] = None

    def move_to_position(self, position: CursorPosition) -> None:
        """Move cursor to a new position."""
        self.position = position
        self.last_moved = datetime.utcnow()
        self.last_updated = datetime.utcnow()

        # Update selection if in selecting state
        if self.state == CursorState.SELECTING and self.selection:
            self.selection.end_position = position
        else:
            # Clear selection and set to single cell
            self.selection = CellRange.from_single_cell(position.address)
            self.selection_type = SelectionType.SINGLE_CELL

    def move_to_address(self, address: str) -> None:
        """Move cursor to a cell address."""
        position = CursorPosition.from_address(address)
        self.move_to_position(position)

    def start_selection(self, start_position: CursorPosition = None) -> None:
        """Start a new selection."""
        start_pos = start_position or self.position
        self.selection = CellRange(
            start_position=start_pos, end_position=start_pos)
        self.selection_type = SelectionType.SINGLE_CELL
        self.state = CursorState.SELECTING
        self.last_updated = datetime.utcnow()

    def extend_selection(self, end_position: CursorPosition) -> None:
        """Extend current selection to end position."""
        if not self.selection:
            self.start_selection()

        self.selection.end_position = end_position
        self.position = end_position

        # Determine selection type
        if self.selection.is_single_cell():
            self.selection_type = SelectionType.SINGLE_CELL
        elif self.selection.is_single_row():
            self.selection_type = SelectionType.ROW
        elif self.selection.is_single_column():
            self.selection_type = SelectionType.COLUMN
        else:
            self.selection_type = SelectionType.RANGE

        self.state = CursorState.SELECTING
        self.last_updated = datetime.utcnow()

    def end_selection(self) -> None:
        """End current selection."""
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def clear_selection(self) -> None:
        """Clear current selection."""
        self.selection = CellRange.from_single_cell(self.position.address)
        self.selection_type = SelectionType.SINGLE_CELL
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def select_range(self, start_address: str, end_address: str) -> None:
        """Select a specific range."""
        self.selection = CellRange.from_addresses(start_address, end_address)

        # Determine selection type
        if self.selection.is_single_cell():
            self.selection_type = SelectionType.SINGLE_CELL
        elif self.selection.is_single_row():
            self.selection_type = SelectionType.ROW
        elif self.selection.is_single_column():
            self.selection_type = SelectionType.COLUMN
        else:
            self.selection_type = SelectionType.RANGE

        # Move cursor to end position
        self.position = self.selection.end_position
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def select_entire_row(self, row: int) -> None:
        """Select an entire row."""
        self.selection = CellRange.from_coordinates(row, 0, row, 25)
        self.selection_type = SelectionType.ROW
        self.position = CursorPosition(row=row, column=0)
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def select_entire_column(self, column: int) -> None:
        """Select an entire column."""
        self.selection = CellRange.from_coordinates(0, column, 99, column)
        self.selection_type = SelectionType.COLUMN
        self.position = CursorPosition(row=0, column=column)
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def start_editing(self, cell_address: str = None) -> None:
        """Start editing a cell."""
        edit_address = cell_address or self.position.address
        self.is_editing = True
        self.editing_cell = edit_address
        self.edit_start_time = datetime.utcnow()
        self.state = CursorState.EDITING
        self.last_updated = datetime.utcnow()

    def end_editing(self) -> None:
        """End cell editing."""
        self.is_editing = False
        self.editing_cell = None
        self.edit_start_time = None
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def set_copying_state(self) -> None:
        """Set cursor to copying state."""
        self.state = CursorState.COPYING
        self.last_updated = datetime.utcnow()

    def set_pasting_state(self) -> None:
        """Set cursor to pasting state."""
        self.state = CursorState.PASTING
        self.last_updated = datetime.utcnow()

    def set_idle_state(self) -> None:
        """Set cursor to idle state."""
        self.state = CursorState.IDLE
        self.last_updated = datetime.utcnow()

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_updated = datetime.utcnow()

    def hide_cursor(self) -> None:
        """Hide cursor from other users."""
        self.is_visible = False
        self.last_updated = datetime.utcnow()

    def show_cursor(self) -> None:
        """Show cursor to other users."""
        self.is_visible = True
        self.last_updated = datetime.utcnow()

    def set_inactive(self) -> None:
        """Mark cursor as inactive."""
        self.is_active = False
        self.is_visible = False
        self.last_updated = datetime.utcnow()

    def set_active(self) -> None:
        """Mark cursor as active."""
        self.is_active = True
        self.is_visible = True
        self.last_updated = datetime.utcnow()

    def get_selected_addresses(self) -> List[str]:
        """Get all selected cell addresses."""
        if not self.selection:
            return [self.position.address]
        return self.selection.get_all_addresses()

    def get_edit_duration(self) -> Optional[float]:
        """Get current edit duration in seconds."""
        if not self.is_editing or not self.edit_start_time:
            return None

        duration = datetime.utcnow() - self.edit_start_time
        return duration.total_seconds()

    def is_inactive(self, timeout_seconds: int = 300) -> bool:
        """Check if cursor has been inactive for too long."""
        if not self.is_active:
            return True

        inactive_duration = datetime.utcnow() - self.last_updated
        return inactive_duration.total_seconds() > timeout_seconds

    def conflicts_with(self, other: 'UserCursor') -> bool:
        """Check if this cursor conflicts with another user's cursor."""
        if not self.selection or not other.selection:
            return self.position == other.position

        return self.selection.intersects_with(other.selection)

    def to_dict(self, include_selection_details: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_color": self.user_color,
            "position": self.position.to_dict(),
            "state": self.state.value,
            "selection_type": self.selection_type.value,
            "last_updated": self.last_updated.isoformat(),
            "last_moved": self.last_moved.isoformat(),
            "is_visible": self.is_visible,
            "is_active": self.is_active,
            "is_editing": self.is_editing,
            "editing_cell": self.editing_cell
        }

        if self.edit_start_time:
            data["edit_start_time"] = self.edit_start_time.isoformat()
            data["edit_duration"] = self.get_edit_duration()

        if self.selection and include_selection_details:
            data["selection"] = self.selection.to_dict()
            data["selected_addresses"] = self.get_selected_addresses()
        elif self.selection:
            data["selection"] = {
                "start_address": self.selection.start_address,
                "end_address": self.selection.end_address,
                "range_notation": self.selection.range_notation
            }

        return data

    @classmethod
    def create_for_user(cls, user_id: str, user_name: str, user_color: str,
                        initial_position: str = "A1") -> 'UserCursor':
        """Create a new user cursor."""
        return cls(
            user_id=user_id,
            user_name=user_name,
            user_color=user_color,
            position=CursorPosition.from_address(initial_position)
        )


class CursorManager(BaseModel):
    """Manages multiple user cursors for collaboration."""

    cursors: Dict[str, UserCursor] = Field(
        default_factory=dict)  # user_id -> UserCursor
    cursor_timeout: int = Field(
        default=300, description="Cursor timeout in seconds")

    def add_cursor(self, cursor: UserCursor) -> None:
        """Add a user cursor."""
        self.cursors[cursor.user_id] = cursor

    def remove_cursor(self, user_id: str) -> bool:
        """Remove a user cursor."""
        if user_id in self.cursors:
            del self.cursors[user_id]
            return True
        return False

    def get_cursor(self, user_id: str) -> Optional[UserCursor]:
        """Get a user's cursor."""
        return self.cursors.get(user_id)

    def update_cursor_position(self, user_id: str, position: CursorPosition) -> bool:
        """Update a user's cursor position."""
        cursor = self.cursors.get(user_id)
        if cursor:
            cursor.move_to_position(position)
            return True
        return False

    def update_cursor_selection(self, user_id: str, start_address: str,
                                end_address: str = None) -> bool:
        """Update a user's selection."""
        cursor = self.cursors.get(user_id)
        if cursor:
            if end_address:
                cursor.select_range(start_address, end_address)
            else:
                cursor.move_to_address(start_address)
            return True
        return False

    def get_active_cursors(self) -> List[UserCursor]:
        """Get all active cursors."""
        return [cursor for cursor in self.cursors.values() if cursor.is_active]

    def get_visible_cursors(self) -> List[UserCursor]:
        """Get all visible cursors."""
        return [cursor for cursor in self.cursors.values() if cursor.is_visible]

    def get_cursors_at_position(self, position: CursorPosition) -> List[UserCursor]:
        """Get all cursors at a specific position."""
        return [
            cursor for cursor in self.cursors.values()
            if cursor.position == position and cursor.is_active
        ]

    def get_cursors_in_range(self, cell_range: CellRange) -> List[UserCursor]:
        """Get all cursors within a range."""
        cursors_in_range = []
        for cursor in self.cursors.values():
            if cursor.is_active and cell_range.contains_position(cursor.position):
                cursors_in_range.append(cursor)
        return cursors_in_range

    def get_conflicting_cursors(self, user_id: str) -> List[UserCursor]:
        """Get cursors that conflict with a specific user's cursor."""
        target_cursor = self.cursors.get(user_id)
        if not target_cursor:
            return []

        conflicts = []
        for cursor in self.cursors.values():
            if (cursor.user_id != user_id and
                cursor.is_active and
                    target_cursor.conflicts_with(cursor)):
                conflicts.append(cursor)

        return conflicts

    def cleanup_inactive_cursors(self) -> List[str]:
        """Remove inactive cursors and return list of removed user IDs."""
        removed_users = []

        for user_id, cursor in list(self.cursors.items()):
            if cursor.is_inactive(self.cursor_timeout):
                removed_users.append(user_id)
                del self.cursors[user_id]

        return removed_users

    def get_cursor_statistics(self) -> Dict[str, Any]:
        """Get statistics about cursors."""
        total_cursors = len(self.cursors)
        active_cursors = len(self.get_active_cursors())
        visible_cursors = len(self.get_visible_cursors())

        # Count by state
        state_counts = {}
        for cursor in self.cursors.values():
            state = cursor.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        # Count by selection type
        selection_counts = {}
        for cursor in self.cursors.values():
            sel_type = cursor.selection_type.value
            selection_counts[sel_type] = selection_counts.get(sel_type, 0) + 1

        # Editing statistics
        editing_count = sum(
            1 for cursor in self.cursors.values() if cursor.is_editing)

        return {
            "total_cursors": total_cursors,
            "active_cursors": active_cursors,
            "visible_cursors": visible_cursors,
            "editing_cursors": editing_count,
            "state_distribution": state_counts,
            "selection_distribution": selection_counts,
            "cursor_timeout": self.cursor_timeout
        }

    def export_cursor_data(self) -> Dict[str, Any]:
        """Export all cursor data."""
        return {
            "cursors": {
                user_id: cursor.to_dict()
                for user_id, cursor in self.cursors.items()
            },
            "statistics": self.get_cursor_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

    def import_cursor_data(self, data: Dict[str, Any]) -> int:
        """Import cursor data and return count of imported cursors."""
        imported_count = 0

        cursor_data = data.get("cursors", {})
        for user_id, cursor_info in cursor_data.items():
            try:
                # Reconstruct cursor from data
                position = CursorPosition(**cursor_info["position"])
                cursor = UserCursor(
                    user_id=user_id,
                    user_name=cursor_info.get("user_name", "Unknown"),
                    user_color=cursor_info.get("user_color", "#000000"),
                    position=position,
                    state=CursorState(cursor_info.get("state", "idle")),
                    selection_type=SelectionType(
                        cursor_info.get("selection_type", "single_cell")),
                    is_visible=cursor_info.get("is_visible", True),
                    is_active=cursor_info.get("is_active", True),
                    is_editing=cursor_info.get("is_editing", False),
                    editing_cell=cursor_info.get("editing_cell")
                )

                # Reconstruct selection if present
                if "selection" in cursor_info and cursor_info["selection"]:
                    sel_data = cursor_info["selection"]
                    cursor.selection = CellRange.from_addresses(
                        sel_data["start_address"],
                        sel_data["end_address"]
                    )

                self.cursors[user_id] = cursor
                imported_count += 1

            except Exception as e:
                # Skip invalid cursor data
                continue

        return imported_count

    def clear_all_cursors(self) -> int:
        """Clear all cursors and return count of removed cursors."""
        removed_count = len(self.cursors)
        self.cursors.clear()
        return removed_count

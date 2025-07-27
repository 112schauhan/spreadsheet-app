from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

from models.operation_model import Operation, OperationType


class HistoryEntryType(str, Enum):
    """Types of history entries."""
    OPERATION = "operation"
    BATCH = "batch"
    SNAPSHOT = "snapshot"
    MERGE = "merge"


class HistoryEntryStatus(str, Enum):
    """Status of history entries."""
    ACTIVE = "active"
    UNDONE = "undone"
    REDONE = "redone"
    MERGED = "merged"
    OBSOLETE = "obsolete"


class HistoryEntry(BaseModel):
    """
    Represents a single entry in the command history.
    Can be an operation, batch of operations, or snapshot.
    """

    # Identity
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: HistoryEntryType

    # Content
    operation: Optional[Operation] = None
    operations: List[Operation] = Field(
        default_factory=list)  # For batch operations
    snapshot_data: Optional[Dict[str, Any]] = None  # For snapshots

    # Metadata
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None

    # Status
    status: HistoryEntryStatus = HistoryEntryStatus.ACTIVE

    # Collaboration info
    session_id: Optional[str] = None
    sequence_number: int = Field(
        default=0, description="Sequence in user's history")
    global_sequence: int = Field(
        default=0, description="Global sequence across all users")

    # Relationships
    # For operations that modify previous ones
    parent_entry_id: Optional[str] = None
    # Operations that modify this one
    child_entry_ids: List[str] = Field(default_factory=list)
    # Related operations from other users
    related_entries: List[str] = Field(default_factory=list)

    # Undo/Redo info
    can_undo: bool = True
    can_redo: bool = False
    # ID of undo operation if this was undone
    undo_entry_id: Optional[str] = None
    # ID of redo operation if this was redone
    redo_entry_id: Optional[str] = None

    # Conflict resolution
    was_transformed: bool = False  # True if operation was transformed due to conflicts
    # Original before transformation
    original_operation: Optional[Operation] = None
    transformation_reason: Optional[str] = None

    @property
    def is_undoable(self) -> bool:
        """Check if this entry can be undone."""
        return (
            self.can_undo and
            self.status == HistoryEntryStatus.ACTIVE and
            self.undo_entry_id is None
        )

    @property
    def is_redoable(self) -> bool:
        """Check if this entry can be redone."""
        return (
            self.can_redo and
            self.status == HistoryEntryStatus.UNDONE and
            self.redo_entry_id is None
        )

    def get_affected_cells(self) -> List[str]:
        """Get list of cells affected by this history entry."""
        affected_cells = []

        if self.operation:
            if self.operation.result and self.operation.result.affected_cells:
                affected_cells.extend(self.operation.result.affected_cells)
            else:
                # Extract from operation target
                target = self.operation.target
                if ':' in target:
                    # Range operation
                    affected_cells.append(target)
                elif target.startswith(('row_', 'column_')):
                    # Row/column operation
                    affected_cells.append(target)
                else:
                    # Single cell
                    affected_cells.append(target)

        for op in self.operations:
            if op.result and op.result.affected_cells:
                affected_cells.extend(op.result.affected_cells)

        return list(set(affected_cells))  # Remove duplicates

    def mark_undone(self, undo_entry_id: str) -> None:
        """Mark this entry as undone."""
        self.status = HistoryEntryStatus.UNDONE
        self.undo_entry_id = undo_entry_id
        self.can_redo = True

    def mark_redone(self, redo_entry_id: str) -> None:
        """Mark this entry as redone."""
        self.status = HistoryEntryStatus.REDONE
        self.redo_entry_id = redo_entry_id
        self.can_redo = False

    def mark_obsolete(self, reason: str = None) -> None:
        """Mark this entry as obsolete."""
        self.status = HistoryEntryStatus.OBSOLETE
        self.can_undo = False
        self.can_redo = False
        if reason:
            self.transformation_reason = reason

    def add_related_entry(self, entry_id: str) -> None:
        """Add a related entry ID."""
        if entry_id not in self.related_entries:
            self.related_entries.append(entry_id)

    def add_child_entry(self, entry_id: str) -> None:
        """Add a child entry ID."""
        if entry_id not in self.child_entry_ids:
            self.child_entry_ids.append(entry_id)

    def set_transformation(self, original_operation: Operation, reason: str) -> None:
        """Mark this entry as transformed."""
        self.was_transformed = True
        self.original_operation = original_operation
        self.transformation_reason = reason

    def to_dict(self, include_operations: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "status": self.status.value,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "global_sequence": self.global_sequence,
            "parent_entry_id": self.parent_entry_id,
            "child_entry_ids": self.child_entry_ids,
            "related_entries": self.related_entries,
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "undo_entry_id": self.undo_entry_id,
            "redo_entry_id": self.redo_entry_id,
            "was_transformed": self.was_transformed,
            "transformation_reason": self.transformation_reason,
            "affected_cells": self.get_affected_cells()
        }

        if include_operations:
            if self.operation:
                data["operation"] = self.operation.to_dict()
            if self.operations:
                data["operations"] = [op.to_dict() for op in self.operations]
            if self.original_operation:
                data["original_operation"] = self.original_operation.to_dict()

        if self.snapshot_data:
            data["snapshot_data"] = self.snapshot_data

        return data

    @classmethod
    def create_operation_entry(cls, operation: Operation, user_id: str,
                               session_id: str = None, description: str = None) -> 'HistoryEntry':
        """Create a history entry for a single operation."""
        return cls(
            entry_type=HistoryEntryType.OPERATION,
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            description=description or f"{operation.operation_type.value} on {operation.target}"
        )

    @classmethod
    def create_batch_entry(cls, operations: List[Operation], user_id: str,
                           session_id: str = None, description: str = None) -> 'HistoryEntry':
        """Create a history entry for a batch of operations."""
        return cls(
            entry_type=HistoryEntryType.BATCH,
            operations=operations,
            user_id=user_id,
            session_id=session_id,
            description=description or f"Batch of {len(operations)} operations"
        )

    @classmethod
    def create_snapshot_entry(cls, snapshot_data: Dict[str, Any], user_id: str,
                              session_id: str = None, description: str = None) -> 'HistoryEntry':
        """Create a history entry for a snapshot."""
        return cls(
            entry_type=HistoryEntryType.SNAPSHOT,
            snapshot_data=snapshot_data,
            user_id=user_id,
            session_id=session_id,
            description=description or "Grid snapshot",
            can_undo=False  # Snapshots typically can't be undone
        )


class CommandHistory(BaseModel):
    """
    Manages command history for undo/redo functionality with collaboration support.
    """

    # Configuration
    max_history_size: int = Field(
        default=100, description="Maximum number of history entries")
    max_user_history: int = Field(
        default=50, description="Maximum history per user")

    # History storage
    entries: Dict[str, HistoryEntry] = Field(
        default_factory=dict)  # entry_id -> HistoryEntry
    user_histories: Dict[str, List[str]] = Field(
        default_factory=dict)  # user_id -> [entry_ids]
    global_history: List[str] = Field(
        default_factory=list)  # Global chronological order

    # Sequence tracking
    global_sequence_counter: int = 0
    user_sequence_counters: Dict[str, int] = Field(default_factory=dict)

    # Undo/Redo stacks per user
    undo_stacks: Dict[str, List[str]] = Field(
        default_factory=dict)  # user_id -> [entry_ids]
    redo_stacks: Dict[str, List[str]] = Field(
        default_factory=dict)  # user_id -> [entry_ids]

    def add_entry(self, entry: HistoryEntry) -> None:
        """Add a new history entry."""
        # Set sequence numbers
        self.global_sequence_counter += 1
        entry.global_sequence = self.global_sequence_counter

        user_id = entry.user_id
        if user_id not in self.user_sequence_counters:
            self.user_sequence_counters[user_id] = 0
        self.user_sequence_counters[user_id] += 1
        entry.sequence_number = self.user_sequence_counters[user_id]

        # Store entry
        self.entries[entry.entry_id] = entry

        # Update global history
        self.global_history.append(entry.entry_id)

        # Update user history
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        self.user_histories[user_id].append(entry.entry_id)

        # Update undo stack (clear redo stack)
        if user_id not in self.undo_stacks:
            self.undo_stacks[user_id] = []
        if entry.can_undo:
            self.undo_stacks[user_id].append(entry.entry_id)

        # Clear redo stack when new operation is added
        self.redo_stacks[user_id] = []

        # Cleanup if needed
        self._cleanup_history()

    def undo_last(self, user_id: str) -> Optional[HistoryEntry]:
        """Undo the last operation for a user."""
        if user_id not in self.undo_stacks or not self.undo_stacks[user_id]:
            return None

        entry_id = self.undo_stacks[user_id].pop()
        entry = self.entries.get(entry_id)

        if not entry or not entry.is_undoable:
            return None

        # Create undo operation
        undo_operation = self._create_undo_operation(entry)
        if undo_operation:
            # Create undo entry
            undo_entry = HistoryEntry.create_operation_entry(
                undo_operation, user_id, entry.session_id, f"Undo {entry.description}"
            )
            undo_entry.parent_entry_id = entry_id

            # Mark original entry as undone
            entry.mark_undone(undo_entry.entry_id)

            # Add to redo stack
            if user_id not in self.redo_stacks:
                self.redo_stacks[user_id] = []
            self.redo_stacks[user_id].append(entry_id)

            # Add undo entry to history
            self.add_entry(undo_entry)

            return entry

        return None

    def redo_last(self, user_id: str) -> Optional[HistoryEntry]:
        """Redo the last undone operation for a user."""
        if user_id not in self.redo_stacks or not self.redo_stacks[user_id]:
            return None

        entry_id = self.redo_stacks[user_id].pop()
        entry = self.entries.get(entry_id)

        if not entry or not entry.is_redoable:
            return None

        # Create redo operation (same as original)
        redo_operation = self._create_redo_operation(entry)
        if redo_operation:
            # Create redo entry
            redo_entry = HistoryEntry.create_operation_entry(
                redo_operation, user_id, entry.session_id, f"Redo {entry.description}"
            )
            redo_entry.parent_entry_id = entry_id

            # Mark original entry as redone
            entry.mark_redone(redo_entry.entry_id)

            # Add back to undo stack
            self.undo_stacks[user_id].append(entry_id)

            # Add redo entry to history
            self.add_entry(redo_entry)

            return entry

        return None

    def can_undo(self, user_id: str) -> bool:
        """Check if user can undo."""
        return (user_id in self.undo_stacks and
                len(self.undo_stacks[user_id]) > 0)

    def can_redo(self, user_id: str) -> bool:
        """Check if user can redo."""
        return (user_id in self.redo_stacks and
                len(self.redo_stacks[user_id]) > 0)

    def get_user_history(self, user_id: str, limit: int = 20) -> List[HistoryEntry]:
        """Get history entries for a user."""
        if user_id not in self.user_histories:
            return []

        entry_ids = self.user_histories[user_id][-limit:]
        return [self.entries[entry_id] for entry_id in entry_ids if entry_id in self.entries]

    def get_global_history(self, limit: int = 50) -> List[HistoryEntry]:
        """Get global history entries."""
        entry_ids = self.global_history[-limit:]
        return [self.entries[entry_id] for entry_id in entry_ids if entry_id in self.entries]

    def get_entries_affecting_cell(self, cell_address: str) -> List[HistoryEntry]:
        """Get all history entries that affect a specific cell."""
        affecting_entries = []

        for entry in self.entries.values():
            if cell_address in entry.get_affected_cells():
                affecting_entries.append(entry)

        return sorted(affecting_entries, key=lambda e: e.global_sequence)

    def get_conflicts_for_entry(self, entry_id: str) -> List[HistoryEntry]:
        """Get entries that conflict with the given entry."""
        entry = self.entries.get(entry_id)
        if not entry:
            return []

        affected_cells = set(entry.get_affected_cells())
        conflicts = []

        for other_entry in self.entries.values():
            if (other_entry.entry_id != entry_id and
                other_entry.user_id != entry.user_id and
                    abs(other_entry.global_sequence - entry.global_sequence) <= 5):  # Within 5 operations

                other_affected = set(other_entry.get_affected_cells())
                if affected_cells.intersection(other_affected):
                    conflicts.append(other_entry)

        return conflicts

    def mark_entries_obsolete(self, entry_ids: List[str], reason: str = None) -> None:
        """Mark multiple entries as obsolete."""
        for entry_id in entry_ids:
            entry = self.entries.get(entry_id)
            if entry:
                entry.mark_obsolete(reason)

                # Remove from undo/redo stacks
                for user_stacks in [self.undo_stacks, self.redo_stacks]:
                    for user_id, stack in user_stacks.items():
                        if entry_id in stack:
                            stack.remove(entry_id)

    def create_snapshot(self, user_id: str, grid_data: Dict[str, Any],
                        description: str = None) -> HistoryEntry:
        """Create a snapshot of the current grid state."""
        snapshot_entry = HistoryEntry.create_snapshot_entry(
            grid_data, user_id, description=description
        )
        self.add_entry(snapshot_entry)
        return snapshot_entry

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the command history."""
        total_entries = len(self.entries)
        user_count = len(self.user_histories)

        # Count by entry type
        type_counts = {}
        status_counts = {}
        for entry in self.entries.values():
            entry_type = entry.entry_type.value
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1

            status = entry.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Undo/redo stack sizes
        undo_counts = {user_id: len(stack)
                       for user_id, stack in self.undo_stacks.items()}
        redo_counts = {user_id: len(stack)
                       for user_id, stack in self.redo_stacks.items()}

        # Transformations
        transformed_count = sum(
            1 for entry in self.entries.values() if entry.was_transformed)

        return {
            "total_entries": total_entries,
            "user_count": user_count,
            "global_sequence": self.global_sequence_counter,
            "entry_types": type_counts,
            "entry_statuses": status_counts,
            "undo_stack_sizes": undo_counts,
            "redo_stack_sizes": redo_counts,
            "transformed_entries": transformed_count,
            "max_history_size": self.max_history_size,
            "current_size": len(self.global_history)
        }

    def _create_undo_operation(self, entry: HistoryEntry) -> Optional[Operation]:
        """Create an undo operation for a history entry."""
        if entry.operation and entry.operation.can_undo():
            return entry.operation.create_undo_operation()
        return None

    def _create_redo_operation(self, entry: HistoryEntry) -> Optional[Operation]:
        """Create a redo operation (same as original) for a history entry."""
        if entry.operation:
            # For redo, we recreate the original operation
            return Operation(
                operation_type=entry.operation.operation_type,
                target=entry.operation.target,
                data=entry.operation.data.copy(),
                user_id=entry.user_id,
                parent_operation_id=entry.operation.operation_id
            )
        return None

    def _cleanup_history(self) -> None:
        """Clean up old history entries when limits are exceeded."""
        # Clean up global history
        if len(self.global_history) > self.max_history_size:
            excess_count = len(self.global_history) - self.max_history_size
            old_entry_ids = self.global_history[:excess_count]

            # Remove from global history
            self.global_history = self.global_history[excess_count:]

            # Mark entries as obsolete
            self.mark_entries_obsolete(
                old_entry_ids, "History size limit exceeded")

        # Clean up per-user histories
        for user_id, user_history in self.user_histories.items():
            if len(user_history) > self.max_user_history:
                excess_count = len(user_history) - self.max_user_history
                old_entry_ids = user_history[:excess_count]

                # Remove from user history
                self.user_histories[user_id] = user_history[excess_count:]

                # Remove from undo/redo stacks
                if user_id in self.undo_stacks:
                    self.undo_stacks[user_id] = [
                        eid for eid in self.undo_stacks[user_id]
                        if eid not in old_entry_ids
                    ]

                if user_id in self.redo_stacks:
                    self.redo_stacks[user_id] = [
                        eid for eid in self.redo_stacks[user_id]
                        if eid not in old_entry_ids
                    ]

    def export_history(self, user_id: str = None) -> Dict[str, Any]:
        """Export history data for backup or analysis."""
        if user_id:
            # Export specific user's history
            user_entries = self.get_user_history(user_id, limit=1000)
            return {
                "user_id": user_id,
                "entries": [entry.to_dict() for entry in user_entries],
                "undo_stack": self.undo_stacks.get(user_id, []),
                "redo_stack": self.redo_stacks.get(user_id, []),
                "sequence_counter": self.user_sequence_counters.get(user_id, 0),
                "export_timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Export all history
            return {
                "entries": {eid: entry.to_dict() for eid, entry in self.entries.items()},
                "user_histories": self.user_histories,
                "global_history": self.global_history,
                "undo_stacks": self.undo_stacks,
                "redo_stacks": self.redo_stacks,
                "global_sequence": self.global_sequence_counter,
                "user_sequences": self.user_sequence_counters,
                "statistics": self.get_statistics(),
                "export_timestamp": datetime.utcnow().isoformat()
            }

    def clear_user_history(self, user_id: str) -> int:
        """Clear history for a specific user."""
        removed_count = 0

        if user_id in self.user_histories:
            user_entry_ids = self.user_histories[user_id]

            # Mark entries as obsolete
            self.mark_entries_obsolete(
                user_entry_ids, f"User {user_id} history cleared")

            # Clear user data
            removed_count = len(user_entry_ids)
            del self.user_histories[user_id]

            if user_id in self.undo_stacks:
                del self.undo_stacks[user_id]
            if user_id in self.redo_stacks:
                del self.redo_stacks[user_id]
            if user_id in self.user_sequence_counters:
                del self.user_sequence_counters[user_id]

        return removed_count

    def clear_all_history(self) -> int:
        """Clear all history."""
        removed_count = len(self.entries)

        # Clear all data structures
        self.entries.clear()
        self.user_histories.clear()
        self.global_history.clear()
        self.undo_stacks.clear()
        self.redo_stacks.clear()
        self.user_sequence_counters.clear()
        self.global_sequence_counter = 0

        return removed_count

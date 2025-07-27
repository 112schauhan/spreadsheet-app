"""
Operation and command models for undo/redo functionality and operation tracking.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class OperationType(str, Enum):
    """Types of operations that can be performed on the spreadsheet."""
    
    # Cell operations
    CELL_UPDATE = "cell_update"
    CELL_DELETE = "cell_delete"
    CELL_FORMAT = "cell_format"
    
    # Range operations
    RANGE_UPDATE = "range_update"
    RANGE_DELETE = "range_delete"
    RANGE_COPY = "range_copy"
    RANGE_PASTE = "range_paste"
    RANGE_FORMAT = "range_format"
    
    # Grid structure operations
    INSERT_ROW = "insert_row"
    DELETE_ROW = "delete_row"
    INSERT_COLUMN = "insert_column"
    DELETE_COLUMN = "delete_column"
    
    # Sorting operations
    SORT_COLUMN = "sort_column"
    SORT_RANGE = "sort_range"
    
    # Formula operations
    FORMULA_UPDATE = "formula_update"
    FORMULA_DELETE = "formula_delete"
    
    # Collaborative operations
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"


class OperationStatus(str, Enum):
    """Status of an operation."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationResult(BaseModel):
    """Result of an operation execution."""
    
    success: bool
    message: Optional[str] = None
    error_code: Optional[str] = None
    affected_cells: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Operation(BaseModel):
    """
    Represents a single operation that can be performed on the spreadsheet.
    Used for undo/redo functionality and operation tracking.
    """
    
    # Operation identification
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType
    
    # Operation data
    target: str  # Cell address, range, or identifier
    data: Dict[str, Any] = Field(default_factory=dict)
    previous_data: Optional[Dict[str, Any]] = None  # For undo operations
    
    # Execution info
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: OperationStatus = OperationStatus.PENDING
    
    # Collaboration info
    version: int = Field(default=1)
    parent_operation_id: Optional[str] = None  # For dependent operations
    
    # Results
    result: Optional[OperationResult] = None
    
    def set_previous_data(self, data: Dict[str, Any]) -> None:
        """Set the previous state data for undo operations."""
        self.previous_data = data
    
    def mark_completed(self, result: OperationResult) -> None:
        """Mark operation as completed with result."""
        self.status = OperationStatus.COMPLETED
        self.result = result
    
    def mark_failed(self, error_message: str, error_code: str = None) -> None:
        """Mark operation as failed."""
        self.status = OperationStatus.FAILED
        self.result = OperationResult(
            success=False,
            message=error_message,
            error_code=error_code
        )
    
    def can_undo(self) -> bool:
        """Check if this operation can be undone."""
        return (
            self.status == OperationStatus.COMPLETED and
            self.previous_data is not None and
            self.operation_type in [
                OperationType.CELL_UPDATE,
                OperationType.CELL_DELETE,
                OperationType.CELL_FORMAT,
                OperationType.RANGE_UPDATE,
                OperationType.RANGE_DELETE,
                OperationType.RANGE_FORMAT,
                OperationType.INSERT_ROW,
                OperationType.DELETE_ROW,
                OperationType.INSERT_COLUMN,
                OperationType.DELETE_COLUMN,
                OperationType.SORT_COLUMN,
                OperationType.FORMULA_UPDATE,
                OperationType.FORMULA_DELETE
            ]
        )
    
    def create_undo_operation(self) -> 'Operation':
        """Create the inverse operation for undo."""
        if not self.can_undo():
            raise ValueError("Operation cannot be undone")
        
        # Determine the undo operation type
        undo_type_map = {
            OperationType.CELL_UPDATE: OperationType.CELL_UPDATE,
            OperationType.CELL_DELETE: OperationType.CELL_UPDATE,
            OperationType.CELL_FORMAT: OperationType.CELL_FORMAT,
            OperationType.RANGE_UPDATE: OperationType.RANGE_UPDATE,
            OperationType.RANGE_DELETE: OperationType.RANGE_UPDATE,
            OperationType.RANGE_FORMAT: OperationType.RANGE_FORMAT,
            OperationType.INSERT_ROW: OperationType.DELETE_ROW,
            OperationType.DELETE_ROW: OperationType.INSERT_ROW,
            OperationType.INSERT_COLUMN: OperationType.DELETE_COLUMN,
            OperationType.DELETE_COLUMN: OperationType.INSERT_COLUMN,
            OperationType.SORT_COLUMN: OperationType.RANGE_UPDATE,
            OperationType.FORMULA_UPDATE: OperationType.FORMULA_UPDATE,
            OperationType.FORMULA_DELETE: OperationType.FORMULA_UPDATE
        }
        
        undo_type = undo_type_map.get(self.operation_type)
        if not undo_type:
            raise ValueError(f"No undo operation defined for {self.operation_type}")
        
        return Operation(
            operation_type=undo_type,
            target=self.target,
            data=self.previous_data.copy(),
            user_id=self.user_id,
            parent_operation_id=self.operation_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary for JSON serialization."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "target": self.target,
            "data": self.data,
            "previous_data": self.previous_data,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "version": self.version,
            "parent_operation_id": self.parent_operation_id,
            "result": self.result.dict() if self.result else None
        }
    
    @classmethod
    def create_cell_update(cls, address: str, new_value: Any, old_value: Any, 
                          user_id: str) -> 'Operation':
        """Create a cell update operation."""
        op = cls(
            operation_type=OperationType.CELL_UPDATE,
            target=address,
            data={"value": new_value},
            user_id=user_id
        )
        op.set_previous_data({"value": old_value})
        return op
    
    @classmethod
    def create_cell_format(cls, address: str, new_format: Dict[str, Any], 
                          old_format: Dict[str, Any], user_id: str) -> 'Operation':
        """Create a cell formatting operation."""
        op = cls(
            operation_type=OperationType.CELL_FORMAT,
            target=address,
            data={"formatting": new_format},
            user_id=user_id
        )
        op.set_previous_data({"formatting": old_format})
        return op
    
    @classmethod
    def create_row_operation(cls, operation_type: OperationType, row_index: int, 
                           user_id: str, row_data: Dict[str, Any] = None) -> 'Operation':
        """Create a row insert/delete operation."""
        return cls(
            operation_type=operation_type,
            target=f"row_{row_index}",
            data={"row_index": row_index, "row_data": row_data or {}},
            user_id=user_id
        )
    
    @classmethod
    def create_column_operation(cls, operation_type: OperationType, column_index: int, 
                              user_id: str, column_data: Dict[str, Any] = None) -> 'Operation':
        """Create a column insert/delete operation."""
        return cls(
            operation_type=operation_type,
            target=f"column_{column_index}",
            data={"column_index": column_index, "column_data": column_data or {}},
            user_id=user_id
        )
    
    @classmethod
    def create_sort_operation(cls, column_index: int, ascending: bool, user_id: str, 
                            original_data: List[Dict[str, Any]]) -> 'Operation':
        """Create a sort operation."""
        op = cls(
            operation_type=OperationType.SORT_COLUMN,
            target=f"column_{column_index}",
            data={"column_index": column_index, "ascending": ascending},
            user_id=user_id
        )
        op.set_previous_data({"original_data": original_data})
        return op
    
    @classmethod
    def create_range_operation(cls, operation_type: OperationType, start_address: str, 
                             end_address: str, data: Dict[str, Any], user_id: str) -> 'Operation':
        """Create a range operation."""
        return cls(
            operation_type=operation_type,
            target=f"{start_address}:{end_address}",
            data=data,
            user_id=user_id
        )


class BatchOperation(BaseModel):
    """
    Represents a batch of operations that should be executed together.
    Used for complex operations that involve multiple steps.
    """
    
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operations: List[Operation] = Field(default_factory=list)
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    
    # Execution tracking
    completed_operations: List[str] = Field(default_factory=list)
    failed_operations: List[str] = Field(default_factory=list)
    
    def add_operation(self, operation: Operation) -> None:
        """Add an operation to the batch."""
        self.operations.append(operation)
    
    def is_completed(self) -> bool:
        """Check if all operations in the batch are completed."""
        return len(self.completed_operations) == len(self.operations)
    
    def has_failures(self) -> bool:
        """Check if any operations in the batch failed."""
        return len(self.failed_operations) > 0
    
    def mark_operation_completed(self, operation_id: str) -> None:
        """Mark an operation as completed."""
        if operation_id not in self.completed_operations:
            self.completed_operations.append(operation_id)
    
    def mark_operation_failed(self, operation_id: str) -> None:
        """Mark an operation as failed."""
        if operation_id not in self.failed_operations:
            self.failed_operations.append(operation_id)
    
    def get_success_rate(self) -> float:
        """Get the success rate of operations in the batch."""
        if not self.operations:
            return 1.0
        
        completed = len(self.completed_operations)
        total = len(self.operations)
        return completed / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch operation to dictionary."""
        return {
            "batch_id": self.batch_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "operations": [op.to_dict() for op in self.operations],
            "completed_operations": self.completed_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.get_success_rate()
        }


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving operation conflicts."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MANUAL_RESOLUTION = "manual_resolution"
    MERGE_CHANGES = "merge_changes"


class OperationConflict(BaseModel):
    """Represents a conflict between operations."""
    
    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_operation: Operation
    conflicting_operation: Operation
    conflict_type: str  # "concurrent_edit", "dependency_violation", etc.
    
    # Resolution
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved_operation: Optional[Operation] = None
    resolution_timestamp: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    def resolve(self, strategy: ConflictResolutionStrategy, 
               resolved_operation: Operation, resolved_by: str) -> None:
        """Resolve the conflict with the given strategy."""
        self.resolution_strategy = strategy
        self.resolved_operation = resolved_operation
        self.resolution_timestamp = datetime.utcnow()
        self.resolved_by = resolved_by
    
    def is_resolved(self) -> bool:
        """Check if the conflict has been resolved."""
        return self.resolved_operation is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conflict to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "original_operation": self.original_operation.to_dict(),
            "conflicting_operation": self.conflicting_operation.to_dict(),
            "conflict_type": self.conflict_type,
            "resolution_strategy": (
                self.resolution_strategy.value 
                if self.resolution_strategy else None
            ),
            "resolved_operation": (
                self.resolved_operation.to_dict() 
                if self.resolved_operation else None
            ),
            "resolution_timestamp": (
                self.resolution_timestamp.isoformat() 
                if self.resolution_timestamp else None
            ),
            "resolved_by": self.resolved_by
        }


class OperationQueue(BaseModel):
    """Queue for managing pending operations."""
    
    queue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operations: List[Operation] = Field(default_factory=list)
    processing: List[str] = Field(default_factory=list)  # Operation IDs being processed
    
    def enqueue(self, operation: Operation) -> None:
        """Add operation to the queue."""
        self.operations.append(operation)
    
    def dequeue(self) -> Optional[Operation]:
        """Get the next operation from the queue."""
        for operation in self.operations:
            if (operation.operation_id not in self.processing and 
                operation.status == OperationStatus.PENDING):
                self.processing.append(operation.operation_id)
                return operation
        return None
    
    def mark_completed(self, operation_id: str) -> None:
        """Mark operation as completed and remove from processing."""
        if operation_id in self.processing:
            self.processing.remove(operation_id)
        
        # Remove completed operation from queue
        self.operations = [
            op for op in self.operations 
            if op.operation_id != operation_id
        ]
    
    def mark_failed(self, operation_id: str) -> None:
        """Mark operation as failed and remove from processing."""
        if operation_id in self.processing:
            self.processing.remove(operation_id)
        
        # Update operation status
        for operation in self.operations:
            if operation.operation_id == operation_id:
                operation.status = OperationStatus.FAILED
                break
    
    def get_pending_count(self) -> int:
        """Get number of pending operations."""
        return len([
            op for op in self.operations 
            if op.status == OperationStatus.PENDING
        ])
    
    def get_processing_count(self) -> int:
        """Get number of operations being processed."""
        return len(self.processing)
    
    def clear_failed_operations(self) -> int:
        """Remove failed operations and return count removed."""
        initial_count = len(self.operations)
        self.operations = [
            op for op in self.operations 
            if op.status != OperationStatus.FAILED
        ]
        return initial_count - len(self.operations)
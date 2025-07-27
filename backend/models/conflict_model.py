"""
Conflict resolution models for handling simultaneous edits in real-time collaboration.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

from models.operation_model import Operation, OperationType


class ConflictType(str, Enum):
    """Types of conflicts that can occur."""
    CONCURRENT_EDIT = "concurrent_edit"
    DEPENDENCY_VIOLATION = "dependency_violation"
    FORMULA_CIRCULAR_REFERENCE = "formula_circular_reference"
    STRUCTURAL_CHANGE = "structural_change"
    PERMISSION_CONFLICT = "permission_conflict"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    VERSION_MISMATCH = "version_mismatch"


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictStatus(str, Enum):
    """Status of conflict resolution."""
    DETECTED = "detected"
    ANALYZING = "analyzing"
    RESOLVED_AUTO = "resolved_auto"
    RESOLVED_MANUAL = "resolved_manual"
    ESCALATED = "escalated"
    FAILED = "failed"


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE_CHANGES = "merge_changes"
    MANUAL_RESOLUTION = "manual_resolution"
    OPERATIONAL_TRANSFORM = "operational_transform"
    REVERT_TO_SNAPSHOT = "revert_to_snapshot"
    USER_PRIORITY = "user_priority"


class ConflictEvidence(BaseModel):
    """Evidence of a conflict between operations."""
    
    evidence_type: str
    description: str
    affected_cells: List[str] = Field(default_factory=list)
    operation_ids: List[str] = Field(default_factory=list)
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evidence_type": self.evidence_type,
            "description": self.description,
            "affected_cells": self.affected_cells,
            "operation_ids": self.operation_ids,
            "severity_score": self.severity_score
        }


class ConflictResolution(BaseModel):
    """Details of how a conflict was resolved."""
    
    strategy: ConflictResolutionStrategy
    winning_operation_id: Optional[str] = None
    losing_operation_ids: List[str] = Field(default_factory=list)
    transformed_operations: List[str] = Field(default_factory=list)
    resolution_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    resolved_by: Optional[str] = None  # User ID or "system"
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    resolution_time_ms: Optional[float] = None
    
    # Quality metrics
    user_satisfaction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    data_loss_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def set_resolution_time(self, start_time: datetime) -> None:
        """Set resolution time based on start time."""
        duration = datetime.utcnow() - start_time
        self.resolution_time_ms = duration.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy.value,
            "winning_operation_id": self.winning_operation_id,
            "losing_operation_ids": self.losing_operation_ids,
            "transformed_operations": self.transformed_operations,
            "resolution_data": self.resolution_data,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat(),
            "resolution_time_ms": self.resolution_time_ms,
            "user_satisfaction": self.user_satisfaction,
            "data_loss_score": self.data_loss_score
        }


class Conflict(BaseModel):
    """
    Represents a conflict between operations in the collaborative spreadsheet.
    """
    
    # Identity
    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    
    # Conflicting operations
    primary_operation: Operation
    conflicting_operations: List[Operation] = Field(default_factory=list)
    
    # Conflict details
    affected_cells: Set[str] = Field(default_factory=set)
    conflict_evidence: List[ConflictEvidence] = Field(default_factory=list)
    
    # Status and resolution
    status: ConflictStatus = ConflictStatus.DETECTED
    resolution: Optional[ConflictResolution] = None
    
    # Timing
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    detection_latency_ms: Optional[float] = None
    
    # Context
    grid_version: int = Field(default=0)
    user_sessions: List[str] = Field(default_factory=list)
    
    # Auto-resolution settings
    auto_resolution_enabled: bool = True
    max_auto_resolution_time_ms: float = Field(default=5000.0)  # 5 seconds
    
    @property
    def is_resolved(self) -> bool:
        """Check if conflict has been resolved."""
        return self.status in [
            ConflictStatus.RESOLVED_AUTO,
            ConflictStatus.RESOLVED_MANUAL
        ]
    
    @property
    def is_active(self) -> bool:
        """Check if conflict is still active."""
        return self.status in [
            ConflictStatus.DETECTED,
            ConflictStatus.ANALYZING
        ]
    
    @property
    def all_operations(self) -> List[Operation]:
        """Get all operations involved in the conflict."""
        return [self.primary_operation] + self.conflicting_operations
    
    @property
    def all_user_ids(self) -> Set[str]:
        """Get all user IDs involved in the conflict."""
        return {op.user_id for op in self.all_operations}
    
    def add_conflicting_operation(self, operation: Operation) -> None:
        """Add another conflicting operation."""
        if operation.operation_id != self.primary_operation.operation_id:
            self.conflicting_operations.append(operation)
            self.affected_cells.update(self._extract_affected_cells(operation))
    
    def add_evidence(self, evidence: ConflictEvidence) -> None:
        """Add evidence of the conflict."""
        self.conflict_evidence.append(evidence)
        
        # Update severity based on evidence
        if evidence.severity_score > 0.8:
            self.severity = ConflictSeverity.CRITICAL
        elif evidence.severity_score > 0.6:
            self.severity = ConflictSeverity.HIGH
        elif evidence.severity_score > 0.3:
            self.severity = ConflictSeverity.MEDIUM
    
    def set_analyzing(self) -> None:
        """Mark conflict as being analyzed."""
        self.status = ConflictStatus.ANALYZING
    
    def set_resolved(self, resolution: ConflictResolution) -> None:
        """Mark conflict as resolved."""
        self.resolution = resolution
        
        if resolution.resolved_by == "system":
            self.status = ConflictStatus.RESOLVED_AUTO
        else:
            self.status = ConflictStatus.RESOLVED_MANUAL
    
    def set_escalated(self) -> None:
        """Mark conflict as escalated for manual resolution."""
        self.status = ConflictStatus.ESCALATED
    
    def set_failed(self) -> None:
        """Mark conflict resolution as failed."""
        self.status = ConflictStatus.FAILED
    
    def calculate_complexity_score(self) -> float:
        """Calculate complexity score for the conflict."""
        base_score = len(self.conflicting_operations) * 0.1
        
        # Add points for different conflict types
        type_scores = {
            ConflictType.CONCURRENT_EDIT: 0.2,
            ConflictType.DEPENDENCY_VIOLATION: 0.4,
            ConflictType.FORMULA_CIRCULAR_REFERENCE: 0.6,
            ConflictType.STRUCTURAL_CHANGE: 0.5,
            ConflictType.PERMISSION_CONFLICT: 0.3,
            ConflictType.DATA_TYPE_MISMATCH: 0.3,
            ConflictType.VERSION_MISMATCH: 0.4
        }
        
        base_score += type_scores.get(self.conflict_type, 0.2)
        
        # Add points for multiple users
        base_score += len(self.all_user_ids) * 0.1
        
        # Add points for affected cells
        base_score += len(self.affected_cells) * 0.05
        
        # Add points for evidence
        if self.conflict_evidence:
            avg_evidence_score = sum(e.severity_score for e in self.conflict_evidence) / len(self.conflict_evidence)
            base_score += avg_evidence_score * 0.3
        
        return min(1.0, base_score)  # Cap at 1.0
    
    def should_auto_resolve(self) -> bool:
        """Determine if conflict should be auto-resolved."""
        if not self.auto_resolution_enabled:
            return False
        
        complexity = self.calculate_complexity_score()
        
        # Auto-resolve simple conflicts
        if complexity < 0.5:
            return True
        
        # Auto-resolve if only two users and concurrent edits
        if (len(self.all_user_ids) == 2 and 
            self.conflict_type == ConflictType.CONCURRENT_EDIT):
            return True
        
        return False
    
    def get_recommended_strategy(self) -> ConflictResolutionStrategy:
        """Get recommended resolution strategy."""
        if self.conflict_type == ConflictType.CONCURRENT_EDIT:
            if len(self.all_user_ids) == 2:
                return ConflictResolutionStrategy.OPERATIONAL_TRANSFORM
            else:
                return ConflictResolutionStrategy.LAST_WRITE_WINS
        
        elif self.conflict_type == ConflictType.DEPENDENCY_VIOLATION:
            return ConflictResolutionStrategy.REVERT_TO_SNAPSHOT
        
        elif self.conflict_type == ConflictType.FORMULA_CIRCULAR_REFERENCE:
            return ConflictResolutionStrategy.REVERT_TO_SNAPSHOT
        
        elif self.conflict_type == ConflictType.STRUCTURAL_CHANGE:
            return ConflictResolutionStrategy.MANUAL_RESOLUTION
        
        elif self.conflict_type == ConflictType.PERMISSION_CONFLICT:
            return ConflictResolutionStrategy.USER_PRIORITY
        
        else:
            return ConflictResolutionStrategy.LAST_WRITE_WINS
    
    def _extract_affected_cells(self, operation: Operation) -> Set[str]:
        """Extract affected cells from an operation."""
        affected = set()
        
        if operation.result and operation.result.affected_cells:
            affected.update(operation.result.affected_cells)
        else:
            # Extract from operation target
            target = operation.target
            if ':' in target:
                # Range
                affected.add(target)
            elif not target.startswith(('row_', 'column_')):
                # Single cell
                affected.add(target)
        
        return affected
    
    def to_dict(self, include_operations: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "affected_cells": list(self.affected_cells),
            "detected_at": self.detected_at.isoformat(),
            "detection_latency_ms": self.detection_latency_ms,
            "grid_version": self.grid_version,
            "user_sessions": self.user_sessions,
            "auto_resolution_enabled": self.auto_resolution_enabled,
            "max_auto_resolution_time_ms": self.max_auto_resolution_time_ms,
            "complexity_score": self.calculate_complexity_score(),
            "recommended_strategy": self.get_recommended_strategy().value,
            "all_user_ids": list(self.all_user_ids),
            "evidence": [e.to_dict() for e in self.conflict_evidence]
        }
        
        if self.resolution:
            data["resolution"] = self.resolution.to_dict()
        
        if include_operations:
            data["primary_operation"] = self.primary_operation.to_dict()
            data["conflicting_operations"] = [op.to_dict() for op in self.conflicting_operations]
        else:
            data["primary_operation_id"] = self.primary_operation.operation_id
            data["conflicting_operation_ids"] = [op.operation_id for op in self.conflicting_operations]
        
        return data
    
    @classmethod
    def create_concurrent_edit_conflict(cls, op1: Operation, op2: Operation) -> 'Conflict':
        """Create a concurrent edit conflict."""
        conflict = cls(
            conflict_type=ConflictType.CONCURRENT_EDIT,
            primary_operation=op1
        )
        conflict.add_conflicting_operation(op2)
        
        # Add evidence
        evidence = ConflictEvidence(
            evidence_type="same_target_different_users",
            description=f"Users {op1.user_id} and {op2.user_id} edited the same target",
            affected_cells=[op1.target, op2.target],
            operation_ids=[op1.operation_id, op2.operation_id],
            severity_score=0.6
        )
        conflict.add_evidence(evidence)
        
        return conflict
    
    @classmethod
    def create_dependency_conflict(cls, dependent_op: Operation, dependency_ops: List[Operation]) -> 'Conflict':
        """Create a dependency violation conflict."""
        conflict = cls(
            conflict_type=ConflictType.DEPENDENCY_VIOLATION,
            primary_operation=dependent_op,
            severity=ConflictSeverity.HIGH
        )
        
        for dep_op in dependency_ops:
            conflict.add_conflicting_operation(dep_op)
        
        # Add evidence
        evidence = ConflictEvidence(
            evidence_type="dependency_violation",
            description="Operation depends on cells modified by other operations",
            operation_ids=[op.operation_id for op in [dependent_op] + dependency_ops],
            severity_score=0.8
        )
        conflict.add_evidence(evidence)
        
        return conflict


class ConflictDetector(BaseModel):
    """Detects conflicts between operations."""
    
    detection_window_ms: float = Field(default=10000.0)  # 10 seconds
    max_concurrent_operations: int = Field(default=100)
    
    def detect_conflicts(self, new_operation: Operation, 
                        recent_operations: List[Operation]) -> List[Conflict]:
        """Detect conflicts with recent operations."""
        conflicts = []
        
        for recent_op in recent_operations:
            if self._operations_conflict(new_operation, recent_op):
                conflict_type = self._determine_conflict_type(new_operation, recent_op)
                
                if conflict_type == ConflictType.CONCURRENT_EDIT:
                    conflict = Conflict.create_concurrent_edit_conflict(new_operation, recent_op)
                    conflicts.append(conflict)
        
        return conflicts
    
    def _operations_conflict(self, op1: Operation, op2: Operation) -> bool:
        """Check if two operations conflict."""
        # Same user operations don't conflict
        if op1.user_id == op2.user_id:
            return False
        
        # Check time window
        time_diff = abs((op1.timestamp - op2.timestamp).total_seconds() * 1000)
        if time_diff > self.detection_window_ms:
            return False
        
        # Check for overlapping targets
        return self._targets_overlap(op1.target, op2.target)
    
    def _targets_overlap(self, target1: str, target2: str) -> bool:
        """Check if operation targets overlap."""
        # Simple implementation - exact match
        if target1 == target2:
            return True
        
        # Check if both are cell addresses in the same area
        # More sophisticated implementation would parse ranges
        return False
    
    def _determine_conflict_type(self, op1: Operation, op2: Operation) -> ConflictType:
        """Determine the type of conflict between operations."""
        # Simple heuristics - can be made more sophisticated
        if op1.target == op2.target:
            return ConflictType.CONCURRENT_EDIT
        
        if (op1.operation_type in [OperationType.INSERT_ROW, OperationType.DELETE_ROW] or
            op2.operation_type in [OperationType.INSERT_ROW, OperationType.DELETE_ROW]):
            return ConflictType.STRUCTURAL_CHANGE
        
        return ConflictType.CONCURRENT_EDIT


class ConflictManager(BaseModel):
    """Manages conflict detection, resolution, and tracking."""
    
    active_conflicts: Dict[str, Conflict] = Field(default_factory=dict)
    resolved_conflicts: Dict[str, Conflict] = Field(default_factory=dict)
    detector: ConflictDetector = Field(default_factory=ConflictDetector)
    
    # Configuration
    max_active_conflicts: int = Field(default=50)
    max_resolved_history: int = Field(default=200)
    auto_cleanup_interval_ms: float = Field(default=300000.0)  # 5 minutes
    
    def detect_and_add_conflicts(self, operation: Operation, 
                               recent_operations: List[Operation]) -> List[Conflict]:
        """Detect conflicts for a new operation and add them to active conflicts."""
        detected_conflicts = self.detector.detect_conflicts(operation, recent_operations)
        
        for conflict in detected_conflicts:
            self.active_conflicts[conflict.conflict_id] = conflict
        
        # Cleanup if needed
        self._cleanup_old_conflicts()
        
        return detected_conflicts
    
    def resolve_conflict(self, conflict_id: str, resolution: ConflictResolution) -> bool:
        """Resolve a conflict."""
        if conflict_id not in self.active_conflicts:
            return False
        
        conflict = self.active_conflicts[conflict_id]
        conflict.set_resolved(resolution)
        
        # Move to resolved conflicts
        self.resolved_conflicts[conflict_id] = conflict
        del self.active_conflicts[conflict_id]
        
        return True
    
    def escalate_conflict(self, conflict_id: str) -> bool:
        """Escalate a conflict for manual resolution."""
        if conflict_id not in self.active_conflicts:
            return False
        
        conflict = self.active_conflicts[conflict_id]
        conflict.set_escalated()
        
        return True
    
    def get_conflicts_for_user(self, user_id: str) -> List[Conflict]:
        """Get all active conflicts involving a user."""
        user_conflicts = []
        
        for conflict in self.active_conflicts.values():
            if user_id in conflict.all_user_ids:
                user_conflicts.append(conflict)
        
        return user_conflicts
    
    def get_conflicts_for_cell(self, cell_address: str) -> List[Conflict]:
        """Get all active conflicts affecting a cell."""
        cell_conflicts = []
        
        for conflict in self.active_conflicts.values():
            if cell_address in conflict.affected_cells:
                cell_conflicts.append(conflict)
        
        return cell_conflicts
    
    def get_auto_resolvable_conflicts(self) -> List[Conflict]:
        """Get conflicts that can be auto-resolved."""
        return [
            conflict for conflict in self.active_conflicts.values()
            if conflict.should_auto_resolve() and conflict.is_active
        ]
    
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get conflict management statistics."""
        active_count = len(self.active_conflicts)
        resolved_count = len(self.resolved_conflicts)
        
        # Count by type
        type_counts = {}
        severity_counts = {}
        status_counts = {}
        
        all_conflicts = list(self.active_conflicts.values()) + list(self.resolved_conflicts.values())
        
        for conflict in all_conflicts:
            # Type distribution
            conflict_type = conflict.conflict_type.value
            type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
            
            # Severity distribution
            severity = conflict.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Status distribution
            status = conflict.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Resolution statistics
        auto_resolved = sum(1 for c in self.resolved_conflicts.values() 
                           if c.status == ConflictStatus.RESOLVED_AUTO)
        manual_resolved = sum(1 for c in self.resolved_conflicts.values() 
                             if c.status == ConflictStatus.RESOLVED_MANUAL)
        
        # Average resolution time
        avg_resolution_time = 0
        resolution_times = [
            c.resolution.resolution_time_ms for c in self.resolved_conflicts.values()
            if c.resolution and c.resolution.resolution_time_ms
        ]
        if resolution_times:
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        return {
            "active_conflicts": active_count,
            "resolved_conflicts": resolved_count,
            "total_conflicts": active_count + resolved_count,
            "type_distribution": type_counts,
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "auto_resolved": auto_resolved,
            "manual_resolved": manual_resolved,
            "average_resolution_time_ms": avg_resolution_time,
            "auto_resolution_rate": auto_resolved / max(1, resolved_count),
            "escalated_conflicts": sum(1 for c in self.active_conflicts.values() 
                                     if c.status == ConflictStatus.ESCALATED)
        }
    
    def _cleanup_old_conflicts(self) -> None:
        """Clean up old conflicts to prevent memory issues."""
        # Remove excess active conflicts (oldest first)
        if len(self.active_conflicts) > self.max_active_conflicts:
            sorted_conflicts = sorted(
                self.active_conflicts.items(),
                key=lambda x: x[1].detected_at
            )
            
            excess_count = len(self.active_conflicts) - self.max_active_conflicts
            for i in range(excess_count):
                conflict_id, conflict = sorted_conflicts[i]
                # Move to resolved with failed status
                conflict.set_failed()
                self.resolved_conflicts[conflict_id] = conflict
                del self.active_conflicts[conflict_id]
        
        # Remove excess resolved conflicts
        if len(self.resolved_conflicts) > self.max_resolved_history:
            sorted_resolved = sorted(
                self.resolved_conflicts.items(),
                key=lambda x: x[1].detected_at
            )
            
            excess_count = len(self.resolved_conflicts) - self.max_resolved_history
            for i in range(excess_count):
                conflict_id, _ = sorted_resolved[i]
                del self.resolved_conflicts[conflict_id]
    
    def export_conflict_data(self) -> Dict[str, Any]:
        """Export conflict data for analysis."""
        return {
            "active_conflicts": {
                cid: conflict.to_dict(include_operations=True)
                for cid, conflict in self.active_conflicts.items()
            },
            "resolved_conflicts": {
                cid: conflict.to_dict(include_operations=True)
                for cid, conflict in self.resolved_conflicts.items()
            },
            "statistics": self.get_conflict_statistics(),
            "detector_config": {
                "detection_window_ms": self.detector.detection_window_ms,
                "max_concurrent_operations": self.detector.max_concurrent_operations
            },
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    def clear_all_conflicts(self) -> Dict[str, int]:
        """Clear all conflicts and return counts."""
        active_count = len(self.active_conflicts)
        resolved_count = len(self.resolved_conflicts)
        
        self.active_conflicts.clear()
        self.resolved_conflicts.clear()
        
        return {
            "cleared_active": active_count,
            "cleared_resolved": resolved_count,
            "total_cleared": active_count + resolved_count
        }
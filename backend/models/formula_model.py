from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class FormulaType(str, Enum):
    MATH = "math"
    STATISTICAL = "statistical"
    LOGICAL = "logical"
    TEXT = "text"
    DATE = "date"
    LOOKUP = "lookup"
    CUSTOM = "custom"


class FormulaStatus(str, Enum):
    PENDING = "pending"
    CALCULATING = "calculating"
    COMPLETED = "completed"
    ERROR = "error"
    CIRCULAR_REFERENCE = "circular_reference"


class FormulaDependency(BaseModel):
    """Represents a dependency relationship between cells."""

    source_cell: str = Field(..., description="Cell containing the formula")
    target_cell: str = Field(..., description="Cell referenced by the formula")
    dependency_type: str = Field(
        default="direct", description="Type of dependency")

    @validator('source_cell', 'target_cell')
    def validate_cell_address(cls, v):
        """Validate cell address format."""
        import re
        if not re.match(r'^[A-Z]+\d+$', v.upper()):
            raise ValueError(f'Invalid cell address format: {v}')
        return v.upper()

    def __hash__(self):
        return hash((self.source_cell, self.target_cell))

    def __eq__(self, other):
        if not isinstance(other, FormulaDependency):
            return False
        return (self.source_cell == other.source_cell and
                self.target_cell == other.target_cell)


class FormulaResult(BaseModel):
    """Result of a formula calculation."""

    success: bool
    value: Any = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Metadata
    cell_address: str
    formula: str
    calculation_time: float = Field(
        default=0.0, description="Time taken to calculate in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Dependencies
    dependencies: Set[str] = Field(
        default_factory=set, description="Cells referenced by this formula")
    dependents: Set[str] = Field(
        default_factory=set, description="Cells that depend on this formula")

    # Validation and warnings
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    @validator('dependencies', 'dependents', pre=True)
    def convert_to_set(cls, v):
        """Convert lists to sets if needed."""
        if isinstance(v, list):
            return set(v)
        return v

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion."""
        if suggestion not in self.suggestions:
            self.suggestions.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "value": self.value,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "cell_address": self.cell_address,
            "formula": self.formula,
            "calculation_time": self.calculation_time,
            "timestamp": self.timestamp.isoformat(),
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }


class Formula(BaseModel):
    """
    Represents a spreadsheet formula with its metadata and calculation state.
    """

    # Identity
    formula_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cell_address: str

    # Formula content
    raw_formula: str = Field(...,
                             description="Original formula as entered by user")
    normalized_formula: str = Field(
        default="", description="Normalized formula for calculation")
    formula_type: FormulaType = FormulaType.CUSTOM

    # Calculation state
    status: FormulaStatus = FormulaStatus.PENDING
    last_calculated: Optional[datetime] = None
    calculation_count: int = Field(
        default=0, description="Number of times this formula has been calculated")

    # Results
    current_result: Optional[FormulaResult] = None
    cached_value: Any = None

    # Dependencies
    direct_dependencies: Set[str] = Field(
        default_factory=set, description="Cells directly referenced")
    indirect_dependencies: Set[str] = Field(
        default_factory=set, description="Cells indirectly referenced")
    dependent_cells: Set[str] = Field(
        default_factory=set, description="Cells that depend on this formula")

    # Metadata
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified_by: Optional[str] = None
    last_modified: datetime = Field(default_factory=datetime.utcnow)

    # Validation and error tracking
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    parse_errors: List[str] = Field(default_factory=list)

    # Performance tracking
    average_calculation_time: float = Field(default=0.0)
    max_calculation_time: float = Field(default=0.0)
    total_calculation_time: float = Field(default=0.0)

    @validator('raw_formula')
    def validate_formula_format(cls, v):
        """Validate formula format."""
        if not v.startswith('='):
            raise ValueError('Formula must start with =')
        return v

    @validator('cell_address')
    def validate_cell_address(cls, v):
        """Validate cell address format."""
        import re
        if not re.match(r'^[A-Z]+\d+$', v.upper()):
            raise ValueError(f'Invalid cell address format: {v}')
        return v.upper()

    @validator('direct_dependencies', 'indirect_dependencies', 'dependent_cells', pre=True)
    def convert_to_set(cls, v):
        """Convert lists to sets if needed."""
        if isinstance(v, list):
            return set(v)
        return v

    def add_dependency(self, cell_address: str, is_direct: bool = True) -> None:
        """Add a dependency to this formula."""
        cell_address = cell_address.upper()
        if is_direct:
            self.direct_dependencies.add(cell_address)
        else:
            self.indirect_dependencies.add(cell_address)

    def remove_dependency(self, cell_address: str) -> None:
        """Remove a dependency from this formula."""
        cell_address = cell_address.upper()
        self.direct_dependencies.discard(cell_address)
        self.indirect_dependencies.discard(cell_address)

    def get_all_dependencies(self) -> Set[str]:
        """Get all dependencies (direct and indirect)."""
        return self.direct_dependencies.union(self.indirect_dependencies)

    def add_dependent(self, cell_address: str) -> None:
        """Add a dependent cell."""
        self.dependent_cells.add(cell_address.upper())

    def remove_dependent(self, cell_address: str) -> None:
        """Remove a dependent cell."""
        self.dependent_cells.discard(cell_address.upper())

    def update_calculation_stats(self, calculation_time: float) -> None:
        """Update calculation performance statistics."""
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.average_calculation_time = self.total_calculation_time / self.calculation_count
        self.max_calculation_time = max(
            self.max_calculation_time, calculation_time)
        self.last_calculated = datetime.utcnow()

    def set_result(self, result: FormulaResult) -> None:
        """Set the calculation result."""
        self.current_result = result
        self.cached_value = result.value if result.success else None
        self.status = FormulaStatus.COMPLETED if result.success else FormulaStatus.ERROR

        if result.calculation_time > 0:
            self.update_calculation_stats(result.calculation_time)

    def set_error(self, error_message: str, error_code: str = None) -> None:
        """Set formula as having an error."""
        self.status = FormulaStatus.ERROR
        self.current_result = FormulaResult(
            success=False,
            error_message=error_message,
            error_code=error_code,
            cell_address=self.cell_address,
            formula=self.raw_formula
        )
        self.cached_value = None

    def mark_calculating(self) -> None:
        """Mark formula as currently being calculated."""
        self.status = FormulaStatus.CALCULATING

    def mark_circular_reference(self) -> None:
        """Mark formula as having a circular reference."""
        self.status = FormulaStatus.CIRCULAR_REFERENCE
        self.set_error("Circular reference detected", "CIRCULAR_REF")

    def add_validation_error(self, error: str) -> None:
        """Add a validation error."""
        if error not in self.validation_errors:
            self.validation_errors.append(error)
        self.is_valid = False

    def clear_validation_errors(self) -> None:
        """Clear all validation errors."""
        self.validation_errors.clear()
        self.is_valid = True

    def add_parse_error(self, error: str) -> None:
        """Add a parse error."""
        if error not in self.parse_errors:
            self.parse_errors.append(error)

    def clear_parse_errors(self) -> None:
        """Clear all parse errors."""
        self.parse_errors.clear()

    def is_ready_for_calculation(self) -> bool:
        """Check if formula is ready for calculation."""
        return (
            self.is_valid and
            len(self.parse_errors) == 0 and
            self.status != FormulaStatus.CALCULATING and
            self.status != FormulaStatus.CIRCULAR_REFERENCE
        )

    def needs_recalculation(self, changed_cells: Set[str]) -> bool:
        """Check if formula needs recalculation based on changed cells."""
        if not changed_cells:
            return False

        all_dependencies = self.get_all_dependencies()
        return bool(all_dependencies.intersection(changed_cells))

    def clone_for_address(self, new_address: str) -> 'Formula':
        """Create a copy of this formula for a different address."""
        return Formula(
            cell_address=new_address,
            raw_formula=self.raw_formula,
            normalized_formula=self.normalized_formula,
            formula_type=self.formula_type,
            created_by=self.created_by,
            last_modified_by=self.last_modified_by
        )

    def to_dict(self, include_stats: bool = False) -> Dict[str, Any]:
        """Convert formula to dictionary for JSON serialization."""
        data = {
            "formula_id": self.formula_id,
            "cell_address": self.cell_address,
            "raw_formula": self.raw_formula,
            "normalized_formula": self.normalized_formula,
            "formula_type": self.formula_type.value,
            "status": self.status.value,
            "last_calculated": self.last_calculated.isoformat() if self.last_calculated else None,
            "cached_value": self.cached_value,
            "direct_dependencies": list(self.direct_dependencies),
            "indirect_dependencies": list(self.indirect_dependencies),
            "dependent_cells": list(self.dependent_cells),
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "parse_errors": self.parse_errors,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "last_modified_by": self.last_modified_by,
            "last_modified": self.last_modified.isoformat()
        }

        if self.current_result:
            data["current_result"] = self.current_result.to_dict()

        if include_stats:
            data["performance_stats"] = {
                "calculation_count": self.calculation_count,
                "average_calculation_time": self.average_calculation_time,
                "max_calculation_time": self.max_calculation_time,
                "total_calculation_time": self.total_calculation_time
            }

        return data

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            set: lambda v: list(v)
        }


class FormulaEngine(BaseModel):
    """Container for managing multiple formulas and their relationships."""

    formulas: Dict[str, Formula] = Field(
        default_factory=dict)  # cell_address -> Formula
    dependency_graph: Dict[str, Set[str]] = Field(
        default_factory=dict)  # cell -> dependencies
    calculation_order: List[str] = Field(
        default_factory=list)  # Topologically sorted cells

    def add_formula(self, formula: Formula) -> None:
        """Add a formula to the engine."""
        self.formulas[formula.cell_address] = formula
        self.dependency_graph[formula.cell_address] = formula.get_all_dependencies(
        )
        self._update_calculation_order()

    def remove_formula(self, cell_address: str) -> bool:
        """Remove a formula from the engine."""
        cell_address = cell_address.upper()
        if cell_address in self.formulas:
            del self.formulas[cell_address]
            if cell_address in self.dependency_graph:
                del self.dependency_graph[cell_address]
            self._update_calculation_order()
            return True
        return False

    def get_formula(self, cell_address: str) -> Optional[Formula]:
        """Get formula for a cell."""
        return self.formulas.get(cell_address.upper())

    def get_dependent_formulas(self, cell_address: str) -> List[Formula]:
        """Get all formulas that depend on a cell."""
        cell_address = cell_address.upper()
        dependents = []

        for formula in self.formulas.values():
            if cell_address in formula.get_all_dependencies():
                dependents.append(formula)

        return dependents

    def _update_calculation_order(self) -> None:
        """Update the topological sort order for formula calculation."""
        try:
            self.calculation_order = self._topological_sort()
        except ValueError as e:
            # Circular reference detected
            self.logger.error(f"Circular reference in formulas: {str(e)}")
            # Mark affected formulas
            for formula in self.formulas.values():
                if formula.status != FormulaStatus.ERROR:
                    formula.mark_circular_reference()

    def _topological_sort(self) -> List[str]:
        """Perform topological sort to determine calculation order."""
        in_degree = {}
        graph = {}

        # Initialize
        for cell in self.dependency_graph:
            in_degree[cell] = 0
            graph[cell] = set()

        # Build graph and calculate in-degrees
        for cell, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                if dep in graph:
                    graph[dep].add(cell)
                    in_degree[cell] += 1

        # Kahn's algorithm
        queue = [cell for cell, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            cell = queue.pop(0)
            result.append(cell)

            for dependent in graph[cell]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for circular references
        if len(result) != len(self.dependency_graph):
            remaining_cells = set(self.dependency_graph.keys()) - set(result)
            raise ValueError(
                f"Circular reference detected in cells: {remaining_cells}")

        return result

    def get_calculation_order(self) -> List[str]:
        """Get the order in which formulas should be calculated."""
        return self.calculation_order.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the formula engine."""
        total_formulas = len(self.formulas)
        valid_formulas = sum(1 for f in self.formulas.values() if f.is_valid)
        error_formulas = sum(1 for f in self.formulas.values()
                             if f.status == FormulaStatus.ERROR)

        formula_types = {}
        for formula in self.formulas.values():
            formula_type = formula.formula_type.value
            formula_types[formula_type] = formula_types.get(
                formula_type, 0) + 1

        avg_calc_time = 0
        if self.formulas:
            total_calc_time = sum(
                f.average_calculation_time for f in self.formulas.values())
            avg_calc_time = total_calc_time / len(self.formulas)

        return {
            "total_formulas": total_formulas,
            "valid_formulas": valid_formulas,
            "error_formulas": error_formulas,
            "formula_types": formula_types,
            "average_calculation_time": avg_calc_time,
            "calculation_order_length": len(self.calculation_order)
        }
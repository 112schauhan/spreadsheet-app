"""
Spreadsheet data model for the collaborative spreadsheet application.
Manages the 26x100 grid of cells and grid operations.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field, validator
from collections import defaultdict
import asyncio
import logging

from .cell_model import Cell, CellType

logger = logging.getLogger(__name__)


class GridDimensions(BaseModel):
    """Grid dimensions configuration."""
    rows: int = Field(default=100, ge=1, le=1000)
    columns: int = Field(default=26, ge=1, le=26)
    
    @validator('columns')
    def validate_columns(cls, v):
        """Ensure columns don't exceed A-Z (26 columns)."""
        if v > 26:
            raise ValueError("Maximum 26 columns (A-Z) supported")
        return v


class Spreadsheet(BaseModel):
    """
    Main spreadsheet model containing the grid of cells.
    Handles cell operations, formulas, and grid management.
    """
    
    # Grid configuration
    dimensions: GridDimensions = Field(default_factory=GridDimensions)
    
    # Grid data - using dictionary for sparse storage
    cells: Dict[str, Cell] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)
    
    # Collaboration tracking
    active_users: Set[str] = Field(default_factory=set)
    
    def get_cell(self, row: int, column: int) -> Cell:
        """Get cell at specified position, create if doesn't exist."""
        address = self._get_address(row, column)
        
        if address not in self.cells:
            self.cells[address] = Cell(row=row, column=column)
        
        return self.cells[address]
    
    def get_cell_by_address(self, address: str) -> Cell:
        """Get cell by A1 notation address."""
        row, column = Cell.parse_address(address)
        return self.get_cell(row, column)
    
    def set_cell_value(self, row: int, column: int, value: Any, user_id: str = None) -> Cell:
        """Set cell value and return the updated cell."""
        cell = self.get_cell(row, column)
        cell.update_value(value, user_id)
        
        # Update spreadsheet metadata
        self.last_modified = datetime.utcnow()
        self.version += 1
        
        return cell
    
    def set_cell_by_address(self, address: str, value: Any, user_id: str = None) -> Cell:
        """Set cell value by A1 notation address."""
        row, column = Cell.parse_address(address)
        return self.set_cell_value(row, column, value, user_id)
    
    def get_range(self, start_address: str, end_address: str) -> List[Cell]:
        """Get cells in a range (e.g., A1:C3)."""
        start_row, start_col = Cell.parse_address(start_address)
        end_row, end_col = Cell.parse_address(end_address)
        
        # Ensure proper order
        min_row, max_row = min(start_row, end_row), max(start_row, end_row)
        min_col, max_col = min(start_col, end_col), max(start_col, end_col)
        
        cells = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cells.append(self.get_cell(row, col))
        
        return cells
    
    def get_column(self, column: int) -> List[Cell]:
        """Get all cells in a column."""
        return [self.get_cell(row, column) for row in range(self.dimensions.rows)]
    
    def get_row(self, row: int) -> List[Cell]:
        """Get all cells in a row."""
        return [self.get_cell(row, col) for col in range(self.dimensions.columns)]
    
    def insert_row(self, row_index: int, user_id: str = None) -> bool:
        """Insert a new row at the specified index."""
        if row_index < 0 or row_index >= self.dimensions.rows:
            return False
        
        # Shift existing cells down
        new_cells = {}
        for address, cell in self.cells.items():
            if cell.row >= row_index:
                # Move cell down by one row
                new_row = cell.row + 1
                if new_row < self.dimensions.rows:  # Don't exceed grid bounds
                    new_cell = Cell(
                        row=new_row,
                        column=cell.column,
                        value=cell.value,
                        formula=cell.formula,
                        formatting=cell.formatting,
                        last_modified_by=user_id
                    )
                    new_address = self._get_address(new_row, cell.column)
                    new_cells[new_address] = new_cell
            else:
                new_cells[address] = cell
        
        self.cells = new_cells
        self._update_metadata(user_id)
        return True
    
    def delete_row(self, row_index: int, user_id: str = None) -> bool:
        """Delete a row at the specified index."""
        if row_index < 0 or row_index >= self.dimensions.rows:
            return False
        
        # Remove cells in the deleted row and shift others up
        new_cells = {}
        for address, cell in self.cells.items():
            if cell.row == row_index:
                # Skip deleted row cells
                continue
            elif cell.row > row_index:
                # Move cell up by one row
                new_row = cell.row - 1
                new_cell = Cell(
                    row=new_row,
                    column=cell.column,
                    value=cell.value,
                    formula=cell.formula,
                    formatting=cell.formatting,
                    last_modified_by=user_id
                )
                new_address = self._get_address(new_row, cell.column)
                new_cells[new_address] = new_cell
            else:
                new_cells[address] = cell
        
        self.cells = new_cells
        self._update_metadata(user_id)
        return True
    
    def insert_column(self, column_index: int, user_id: str = None) -> bool:
        """Insert a new column at the specified index."""
        if column_index < 0 or column_index >= self.dimensions.columns:
            return False
        
        # Shift existing cells right
        new_cells = {}
        for address, cell in self.cells.items():
            if cell.column >= column_index:
                # Move cell right by one column
                new_column = cell.column + 1
                if new_column < self.dimensions.columns:  # Don't exceed grid bounds
                    new_cell = Cell(
                        row=cell.row,
                        column=new_column,
                        value=cell.value,
                        formula=cell.formula,
                        formatting=cell.formatting,
                        last_modified_by=user_id
                    )
                    new_address = self._get_address(cell.row, new_column)
                    new_cells[new_address] = new_cell
            else:
                new_cells[address] = cell
        
        self.cells = new_cells
        self._update_metadata(user_id)
        return True
    
    def delete_column(self, column_index: int, user_id: str = None) -> bool:
        """Delete a column at the specified index."""
        if column_index < 0 or column_index >= self.dimensions.columns:
            return False
        
        # Remove cells in the deleted column and shift others left
        new_cells = {}
        for address, cell in self.cells.items():
            if cell.column == column_index:
                # Skip deleted column cells
                continue
            elif cell.column > column_index:
                # Move cell left by one column
                new_column = cell.column - 1
                new_cell = Cell(
                    row=cell.row,
                    column=new_column,
                    value=cell.value,
                    formula=cell.formula,
                    formatting=cell.formatting,
                    last_modified_by=user_id
                )
                new_address = self._get_address(cell.row, new_column)
                new_cells[new_address] = new_cell
            else:
                new_cells[address] = cell
        
        self.cells = new_cells
        self._update_metadata(user_id)
        return True
    
    def sort_column(self, column: int, ascending: bool = True, user_id: str = None) -> bool:
        """Sort data in a column."""
        if column < 0 or column >= self.dimensions.columns:
            return False
        
        # Get all non-empty cells in the column
        column_cells = []
        for row in range(self.dimensions.rows):
            cell = self.get_cell(row, column)
            if not cell.is_empty:
                column_cells.append((row, cell))
        
        if not column_cells:
            return True  # Nothing to sort
        
        # Sort cells by value
        try:
            column_cells.sort(
                key=lambda x: self._get_sort_key(x[1]),
                reverse=not ascending
            )
        except Exception as e:
            logger.error(f"Error sorting column {column}: {str(e)}")
            return False
        
        # Clear the column
        for row in range(self.dimensions.rows):
            address = self._get_address(row, column)
            if address in self.cells:
                del self.cells[address]
        
        # Place sorted cells back
        for new_row, (old_row, cell) in enumerate(column_cells):
            new_cell = Cell(
                row=new_row,
                column=column,
                value=cell.value,
                formula=cell.formula,
                formatting=cell.formatting,
                last_modified_by=user_id
            )
            new_address = self._get_address(new_row, column)
            self.cells[new_address] = new_cell
        
        self._update_metadata(user_id)
        return True
    
    def copy_range(self, start_address: str, end_address: str) -> List[Dict[str, Any]]:
        """Copy a range of cells and return their data."""
        cells = self.get_range(start_address, end_address)
        return [cell.to_dict() for cell in cells]
    
    def paste_range(self, start_address: str, cell_data: List[Dict[str, Any]], user_id: str = None) -> bool:
        """Paste cell data starting at the specified address."""
        try:
            start_row, start_col = Cell.parse_address(start_address)
            
            for i, data in enumerate(cell_data):
                # Calculate target position
                source_row = data.get('row', 0)
                source_col = data.get('column', 0)
                target_row = start_row + (source_row - cell_data[0].get('row', 0))
                target_col = start_col + (source_col - cell_data[0].get('column', 0))
                
                # Check bounds
                if (target_row >= self.dimensions.rows or 
                    target_col >= self.dimensions.columns or
                    target_row < 0 or target_col < 0):
                    continue
                
                # Set cell value
                value_data = data.get('value', {})
                raw_value = value_data.get('raw')
                formula = data.get('formula')
                
                cell = self.get_cell(target_row, target_col)
                if formula:
                    cell.update_value(formula, user_id)
                else:
                    cell.update_value(raw_value, user_id)
                
                # Apply formatting if present
                formatting_data = data.get('formatting', {})
                if formatting_data:
                    for key, value in formatting_data.items():
                        if hasattr(cell.formatting, key):
                            setattr(cell.formatting, key, value)
            
            self._update_metadata(user_id)
            return True
            
        except Exception as e:
            logger.error(f"Error pasting range: {str(e)}")
            return False
    
    def clear_range(self, start_address: str, end_address: str, user_id: str = None) -> bool:
        """Clear cells in a range."""
        try:
            cells = self.get_range(start_address, end_address)
            for cell in cells:
                cell.update_value(None, user_id)
            
            self._update_metadata(user_id)
            return True
            
        except Exception as e:
            logger.error(f"Error clearing range: {str(e)}")
            return False
    
    def get_formula_dependencies(self, address: str) -> Set[str]:
        """Get all cell addresses that this formula depends on."""
        cell = self.get_cell_by_address(address)
        dependencies = set()
        
        if cell.is_formula and cell.formula:
            # Simple regex to find cell references (A1, B2, etc.)
            import re
            pattern = r'\b([A-Z]+\d+)\b'
            matches = re.findall(pattern, cell.formula.upper())
            dependencies.update(matches)
        
        return dependencies
    
    def get_dependent_cells(self, address: str) -> Set[str]:
        """Get all cells that depend on this cell."""
        dependents = set()
        
        for cell_address, cell in self.cells.items():
            if cell.is_formula:
                dependencies = self.get_formula_dependencies(cell_address)
                if address.upper() in dependencies:
                    dependents.add(cell_address)
        
        return dependents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spreadsheet to dictionary for JSON serialization."""
        return {
            "dimensions": self.dimensions.dict(),
            "cells": {address: cell.to_dict() for address, cell in self.cells.items()},
            "metadata": {
                "created_at": self.created_at.isoformat(),
                "last_modified": self.last_modified.isoformat(),
                "version": self.version
            },
            "active_users": list(self.active_users)
        }
    
    def get_non_empty_cells(self) -> Dict[str, Cell]:
        """Get all non-empty cells."""
        return {addr: cell for addr, cell in self.cells.items() if not cell.is_empty}
    
    def get_cell_count(self) -> int:
        """Get total number of cells with data."""
        return len(self.get_non_empty_cells())
    
    def _get_address(self, row: int, column: int) -> str:
        """Get cell address in A1 notation."""
        column_letter = chr(ord('A') + column)
        return f"{column_letter}{row + 1}"
    
    def _get_sort_key(self, cell: Cell) -> Any:
        """Get sort key for a cell value."""
        if cell.value.cell_type == CellType.NUMBER:
            return (0, float(cell.value.raw_value or 0))
        elif cell.value.cell_type == CellType.DATE:
            try:
                from dateutil.parser import parse
                return (1, parse(cell.value.display_value))
            except:
                return (3, cell.value.display_value)
        elif cell.value.cell_type == CellType.TEXT:
            return (2, cell.value.display_value.lower())
        else:
            return (3, cell.value.display_value)
    
    def _update_metadata(self, user_id: str = None) -> None:
        """Update spreadsheet metadata."""
        self.last_modified = datetime.utcnow()
        self.version += 1
        if user_id:
            self.active_users.add(user_id)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
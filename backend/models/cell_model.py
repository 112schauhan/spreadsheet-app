from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Optional, Any, Dict, Union, List
from datetime import datetime
import re


class CellType(str, Enum):
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    FORMULA = "formula"
    EMPTY = "empty"
    ERROR = "error"


class CellValue(BaseModel):
    raw_value: Any = None
    display_value: str = ""
    cell_type: CellType = CellType.EMPTY

    @validator('cell_type', pre=True)
    def validate_cell_type(cls, value):
        if isinstance(value, str):
            return CellType(value.lower())
        return value

    def __str__(self):
        return self_display_value or ""

    def __repr__(self):
        return f"CellValue(type={self.cell_type}, value='{self.display_value}')"


class CellFormatting(BaseModel):
    font_family: Optional[str] = None
    font_size: Optional[int] = None
    bold: Optional[bool] = False
    italic: Optional[bool] = False
    underline: Optional[bool] = False
    text_color: Optional[str] = None
    background_color: Optional[str] = None
    text_align: Optional[str] = None
    vertical_align: Optional[str] = None

    @validator('text_color', 'background_color')
    def validate_color(cls, value):
        if value is not None and not re.match(r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$', value):
            raise ValueError("Color must be a valid hex code")
        return value


class Cell(BaseModel):
    row: int = Field(..., ge=0, le=99, description="Row index of the cell")
    column: int = Field(..., ge=0, le=25,
                        description="Column index of the cell")

    value: CellValue = Field(default_factory=CellValue,
                             description="Value of the cell")
    formatting: CellFormatting = Field(
        default_factory=CellFormatting, description="Formatting of the cell")
    last_modified: datetime = Field(
        default_factory=datetime.utcnow, description="Last modified timestamp of the cell")
    last_modified_by: Optional[str] = None

    version: int = Field(
        default=1, description="Version for conflict resolution")

    validation_errors: List[str] = Field(default_factory=list)
    is_locked: bool = False

    @property
    def address(self) -> str:
        column_letter = chr(ord('A') + self.column)
        return f"{column_letter}{self.row + 1}"

    @property
    def is_empty(self) -> bool:
        return self.value.cell_type == CellType.EMPTY

    @property
    def is_formula(self) -> bool:
        return self.value.cell_type == CellType.FORMULA and self.formula is not None

    @property
    def is_error(self) -> bool:
        return self.value.cell_type == CellType.ERROR

    @classmethod
    def from_address(cls, address: str, **kwargs) -> 'Cell':
        row, column = cls.parse_address(address)
        return cls(row=row, column=column, **kwargs)

    @staticmethod
    def parse_address(address: str) -> tuple[int, int]:
        match = re.match(r'^([A-Z]+)(\d+)$', address.upper())
        if not match:
            raise ValueError(f"Invalid cell address: {address}")

        column_letters, row_str = match.groups()

        column = 0
        for char in column_letters:
            column = column * 26 + (ord(char) - ord('A'))

        row = int(row_str) - 1

        if row < 0 or row > 99:
            raise ValueError(f"Row must be between 1 and 100: {row + 1}")
        if column < 0 or column > 25:
            raise ValueError(
                f"Column must be between A and Z: {column_letters}")

        return row, column

    def update_value(self, new_value: Any, user_id: str = None) -> None:
        """Update cell value and metadata."""
        # Determine cell type and format value
        if new_value is None or str(new_value).strip() == "":
            self.value = CellValue(
                raw_value=None,
                display_value="",
                cell_type=CellType.EMPTY
            )
            self.formula = None
        elif isinstance(new_value, str) and new_value.startswith('='):
            # Formula
            self.formula = new_value
            self.value = CellValue(
                raw_value=new_value,
                display_value=new_value,
                cell_type=CellType.FORMULA
            )
        else:
            # Detect and set appropriate type
            cell_type, formatted_value = self._detect_cell_type(new_value)
            self.value = CellValue(
                raw_value=new_value,
                display_value=str(formatted_value),
                cell_type=cell_type
            )
            self.formula = None

        # Update metadata
        self.last_modified = datetime.utcnow()
        self.last_modified_by = user_id
        self.version += 1
        self.validation_errors.clear()

    def _detect_cell_type(self, value: Any) -> tuple[CellType, Any]:
        if isinstance(value, bool):
            return CellType.BOOLEAN, value

        if isinstance(value, (int, float)):
            return CellType.NUMBER, value

        if isinstance(value, datetime):
            return CellType.DATE, value.isoformat()

        try:
            if '.' in str(value) or 'e' in str(value).lower():
                float_val = float(value)
                return CellType.NUMBER, float_val
            else:
                int_val = int(value)
                return CellType.NUMBER, int_val
        except (ValueError, TypeError):
            pass

        try:
            from dateutil.parser import parse
            date_val = parse(str(value))
            return CellType.DATE, date_val.isoformat()
        except (ValueError, TypeError):
            pass

        return CellType.TEXT, str(value)

    def add_validation_error(self, error: str) -> None:
        if error not in self.validation_errors:
            self.validation_errors.append(error)

    def clear_validation_errors(self) -> None:
        self.validation_errors.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "row": self.row,
            "column": self.column,
            "value": {
                "raw": self.value.raw_value,
                "display": self.value.display_value,
                "type": self.value.cell_type.value
            },
            "formula": self.formula,
            "formatting": self.formatting.dict(),
            "metadata": {
                "last_modified": self.last_modified.isoformat(),
                "last_modified_by": self.last_modified_by,
                "version": self.version,
                "is_locked": self.is_locked
            },
            "validation_errors": self.validation_errors
        }

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

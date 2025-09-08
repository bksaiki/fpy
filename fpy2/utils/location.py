"""
This module defines a parse location.
"""

from dataclasses import dataclass

@dataclass
class Location:
    """Parse location: line and column number."""
    source: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __key(self):
        return (
            self.source,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        return self.__key() == other.__key()

    def format(self):
        return f'`{self.source}:{self.start_line}:{self.start_column}'

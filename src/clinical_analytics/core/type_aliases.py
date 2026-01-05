"""Type aliases for consistent type annotations across codebase."""

from typing import Any

# Common dictionary types
ConfigDict = dict[str, Any]
FilterDict = dict[str, Any]
ColumnMapping = dict[str, str]
MetadataDict = dict[str, Any]

# Common optional types
OptionalStr = str | None
OptionalInt = int | None
OptionalFloat = float | None
OptionalDict = dict[str, Any] | None
OptionalList = list[str] | None

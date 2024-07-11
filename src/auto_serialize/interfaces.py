"""
This script contains the definition of the Serializable interface, which must
be implemented by anything serializable.
"""

# We include this because pylint really doesn't like orjson.
# pylint: disable=no-member

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import orjson
import yaml


class Serializable(ABC):
    """
    Anything that is marked as Serializable can be serialized using the
    serialization module.
    """

    def __eq__(self, value: object) -> bool:
        """
        A default implementation of the equality operator for anything that
        implements AutoSerialize. Note that this implementation should always
        be correct, but it is likely to be much slower than a manually
        implemented equality operator.
        """
        # If the objects are of different types, they can't be equal.
        if not isinstance(value, type(self)):
            return False

        # Convert the objects to json byte arrays and compare the byte arrays.
        return self.to_dict() == value.to_dict()

    def __ne__(self, value: object) -> bool:
        """
        A default implementation of the inequality operator for anything that
        implements AutoSerialize. Note that this implementation should always
        be correct, but it is likely to be much slower than a manually
        implemented inequality operator.
        """
        return not self == value

    def __hash__(self) -> int:
        """
        A default implementation of the hash function for anything that
        implements AutoSerialize. Note that this implementation should always
        be correct, but it is likely to be much slower than a manually
        implemented hash function.
        """
        return hash(self.to_json_bytes())

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the object.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> Self:
        """
        Returns an instance of the class from the dictionary representation.
        """

    def to_json_bytes(self) -> bytes:
        """
        Returns a JSON representation of the object.
        """
        return orjson.dumps(self.to_dict())

    def to_json(self) -> str:
        """
        Returns a JSON representation of the object.
        """
        return self.to_json_bytes().decode()

    def to_yaml(self) -> str:
        """
        Returns a YAML representation of the object.
        """
        return yaml.dump(self.to_dict())

    def deepcopy(self) -> Self:
        """
        Returns a deep copy of the object.

        This is internally done by converting the object to and from a json byte
        array.
        """
        return self.from_json_bytes(self.to_json_bytes())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> Self:
        """
        Returns an instance of the class from the JSON representation.
        """
        return cls.from_dict(orjson.loads(data))

    @classmethod
    def from_json(cls, data: str) -> Self:
        """
        Returns an instance of the class from the JSON representation.
        """
        return cls.from_json_bytes(data.encode())

    @classmethod
    def from_yaml(cls, data: str) -> Self:
        """
        Returns an instance of the class from the YAML representation.
        """
        return cls.from_dict(yaml.full_load(data))

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """
        Returns an instance of the class from a file.
        """
        # Make sure the path is a Path object.
        path = Path(path)

        # Check the file extension.
        if path.suffix == ".json":
            # Read the JSON file.
            return cls.from_json_bytes(path.read_bytes())
        if path.suffix == ".yaml":
            # Read the YAML file.
            return cls.from_yaml(path.read_text(encoding="utf-8"))

        raise ValueError(f"Unsupported file extension: {path.suffix}")

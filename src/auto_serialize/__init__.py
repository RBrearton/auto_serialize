"""
This module defines a simple AutoSerialize mixin class that is pretty good at
automagically converting objects to and from JSON and YAML.

This is not a validation library - pydantic is already excellent at that. This
is a library that, generally without any code modification, can figure out how
to serialize and deserialize your objects. Pydantic's focus on rigour and
safety can often make it inconvenient.
"""

from ._auto_serialize import CLASS_NAME, AutoSerialize

__all__ = ["AutoSerialize", "CLASS_NAME"]

__version__ = "1.0.0"
"""
The version of the auto_serialize module.
"""

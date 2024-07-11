"""
This script contains the definition of the AutoSerialize class, which is a mixin
class that allows for automatic serialization/deserialization of objects.
"""

from importlib import import_module
from types import EllipsisType, UnionType
from typing import (
    Any,
    Literal,
    Mapping,
    Self,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np

from .interfaces import Serializable

CLASS_NAME = "class_name"


# A set of all serializable types.
SERIALIZABLE_TYPES = (
    str,
    int,
    float,
    bool,
    np.ndarray,
    list,
    dict,
    tuple,
    Serializable,
    set,
)


def is_optional(type_hint: Any) -> bool:
    """
    A simple function that checks if a field is Optional (thanks for this one,
    StackOverflow).
    """
    return get_origin(type_hint) is Union and type(None) in get_args(type_hint)


def fully_qualified_name(o: Any) -> str:
    """
    A simple function that returns the fully qualified name of a class. Once
    again, inspiration from StackOverflow.
    """
    # Work out if o is a type or an instance.
    if isinstance(o, type):
        class_object = o
    else:
        class_object = o.__class__

    module = class_object.__module__
    if module == "builtins":
        return class_object.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + class_object.__qualname__


def has_appropriate_attr(obj: Any, attr_name: str) -> bool:
    """
    Returns True if the object has an attribute with the name passed as an
    argument, which is appropriate for serialization.
    """
    # First, if the attribute doesn't exist, return False.
    if not hasattr(obj, attr_name):
        return False

    # Get the attribute, so we can run some tests on it.
    attr = getattr(obj, attr_name)

    # If the attribute exists, make sure it isn't callable.
    if callable(attr):
        return False

    # If the attribute exists and is None, return True - we can serialize None.
    if attr is None:
        return True

    # If the attribute exists and is not a serializable type, return False.
    if not isinstance(attr, SERIALIZABLE_TYPES):
        return False

    # If the attribute exists and isn't callable, return True.
    return True


def get_input_type_hints(func: Any) -> dict[str, Any]:
    """
    Returns the type hints for a function, excluding the return type.
    """
    # Get the type hints for the function.
    type_hints = get_type_hints(func)

    # Remove the "return" key from the type hints, if it is present.
    if "return" in type_hints:
        del type_hints["return"]

    return type_hints


class AutoSerialize(Serializable):
    """
    Inherit from the AutoSerialize class to automatically be able to serialize
    and deserialize your classes to and from various data formats.

    Please note that, if your arguments to __init__ are not type hinted, this
    will not work. Also note that, if your arguments to __init__ are custom
    classes, those custom classes must also inherit from AutoSerialize.

    This class depends on your type hints for the __init__ method, and requires
    that your classes follow one of the following conventions:

    Convention 1:
    ```
    class MyClass1(AutoSerialize):
        def __init__(self, arg1: str, arg2: int):
            self.arg1 = arg1
            self.arg2 = arg2
    ```

    Convention 2:
    ```
    class MyClass2(AutoSerialize):
        def __init__(self, arg1: str, arg2: int):
            self._arg1 = arg1
            self._arg2 = arg2

        @property
        def arg1(self) -> str:
            return some_function(self._arg1)

        @property
        def arg2(self) -> int:
            return some_function(self._arg2)
    ```

    Convention 1 is fairly self-explanatory. We use the type hints on the
    __init__ method to figure out that instances of MyClass1 should have an
    attribute called "arg1" that is a string, and an attribute called "arg2"
    that is an integer. Thus, we have enough information to serialize the
    object.

    Convention 2 demonstrates the true algorithm that AutoSerialize is using.
    In this case, we have two properties, arg1 and arg2, that are each a
    function of the protected attributes _arg1 and _arg2. In this case,
    AutoSerialize will serialize the object **based on the protected attributes,
    not the properties**.

    There are more terms and conditions here. First of all, if _arg1 is (for
    some crazy reason) callable, or otherwise not serializable, while the
    property arg1 is serializable, AutoSerialize will serialize arg1, not _arg1.

    Basically, it tries to serialize _arg1 first, and if that fails, it falls
    back to arg1. If neither of those work, it will fail to serialize the
    object.
    """

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return f"{fully_qualified_name(self)}: {self.to_dict()}"

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return self.__repr__()

    def before_serialize(self) -> None:
        """
        This method is called before the object is serialized. It can be
        overridden by subclasses to perform any necessary operations before
        serialization.
        """

    def after_deserialize(self) -> None:
        """
        This method is called after the object is deserialized. It can be
        overridden by subclasses to perform any necessary operations after
        deserialization.
        """

    def to_dict(self, include_class_name: bool = False) -> dict[str, Any]:
        """
        Returns a dictionary representation of the object.
        """
        # Start by calling the before_serialize method.
        self.before_serialize()

        return_dict = {}

        # Get the type hints for this class's __init__ method.
        type_hints = get_input_type_hints(self.__init__)

        # Get the attribute names that we want to serialize.
        for name, attr_name in zip(
            self._get_attr_names_to_pass_to_init(),
            self._get_attr_names_to_serialize(),
        ):
            # Note that now name is the "public facing" name that'll end up in
            # the dictionary, and attr_name is the name of the attribute on the
            # object that we've decided to serialize.
            attr = getattr(self, attr_name)

            # If the attribute is None, we don't want to serialize it.
            if attr is None:
                continue

            # Otherwise, serialize the attribute.
            return_dict[name] = self._serialize_value(attr, type_hints[name])

        # Finally, we add the fully qualified name of the class to the
        # dictionary, if the user has requested it.
        if include_class_name:
            return_dict[CLASS_NAME] = fully_qualified_name(self)

        return return_dict

    def _get_attr_names_to_serialize(self) -> list[str]:
        """
        Returns a list of attribute names that should be serialized.
        """
        # Get the type hints for the __init__ method, ignoring the return type.
        type_hints = get_input_type_hints(self.__init__)

        # The list that we want to return.
        attr_names = []

        # For each key in the type hints, check if we have a protected attribute
        # with the same name. If we do, add it to the list of attributes to
        # serialize.
        for key in type_hints.keys():
            trial_key = f"_{key}"
            if has_appropriate_attr(self, trial_key):
                attr_names.append(trial_key)

            # If we don't have a protected attribute with the same name, check
            # if there's a public attribute with the same name.
            elif has_appropriate_attr(self, key):
                attr_names.append(key)
            else:
                raise AttributeError(
                    "Could not find a suitable attribute for serialization: "
                    f"{key}"
                )

        return attr_names

    def _serialize_value(self, value: Any, value_type_hint: Any) -> Any:
        """
        Serializes a value.
        """
        # If the value is an instance of AutoSerialize, we need to call its
        # to_dict method.
        if isinstance(value, AutoSerialize):
            # Check to see if the value's type hint matches the actual type of
            # the value. If it doesn't, we need to add the class name to the
            # dictionary, so that we have enough metadata to deserialize the
            # object (as clearly the type hints will be insufficient).
            include_class_name = value_type_hint != type(value)
            return value.to_dict(include_class_name=include_class_name)

        # If the value is a numpy array, we need to make sure that we serialize
        # it properly.
        if isinstance(value, np.ndarray):
            return value.tolist()

        # If the value is a tuple, we need to serialize each item in the tuple.
        # Any other sequences will be serialized as lists. Most behaviours would
        # not break if this check was removed (because tuples are sequences),
        # but it's nice to explicitly check for tuples.
        if isinstance(value, tuple):
            hint = get_args(value_type_hint)[0]
            return tuple(self._serialize_value(item, hint) for item in value)

        # If the value is a sequence, serialize each item as a list. We need to
        # be a little careful here, as strings are sequences, but we don't want
        # to serialize them as lists of individual characters!!
        if isinstance(value, Sequence) and not isinstance(value, str):
            hint = get_args(value_type_hint)[0]
            return [self._serialize_value(item, hint) for item in value]

        # If the value is a dictionary, (or any other mapping) we need to
        # serialize each key, value pair in the dictionary.
        if isinstance(value, Mapping):
            key_hint, val_hint = get_args(value_type_hint)
            return {
                self._serialize_value(key, key_hint): self._serialize_value(
                    val, val_hint
                )
                for key, val in value.items()
            }

        # If the value is a set, we need to serialize each item in the set as a
        # list. We do this because sets aren't json serializable.
        if isinstance(value, set):
            hint = get_args(value_type_hint)[0]
            return [self._serialize_value(item, hint) for item in value]

        # If execution reaches here, we can just return the value.
        return value

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> Self:
        # Get init's type hints.
        type_hints = get_input_type_hints(cls.__init__)

        # Note that, if, somewhere in our codebase, we have an __init__ method
        # that takes list[BaseClass] as an argument, where BaseClass is
        # Serializable, we need to make sure that we try to deserialize the
        # correct child class. We rely on the to_dict method to provide the
        # child class name when we need it.
        child_class_name = input_dict.get(CLASS_NAME)

        # If the child class name was provided and is not the same as the
        # current class name, we need to call from_dict on the child class.
        if child_class_name is not None:
            if child_class_name != fully_qualified_name(cls):
                # We need to dynamically import the child class.
                child_class = cls._import_class(child_class_name)

                # Call from_dict on the child class.
                return child_class.from_dict(input_dict)

            # Otherwise, remove the class name from the dictionary so that
            # we don't try to deserialize it.
            del input_dict[CLASS_NAME]

        # Make a new dictionary that we're going to pass to the __init__ method.
        kwargs_dict = {}

        # Now we want to populate the kwargs_dict by properly deserializing
        # every value that we've been given.
        for input_key, input_value in input_dict.items():
            kwargs_dict[input_key] = cls._deserialize_value(
                input_value, type_hints[input_key]
            )

        # Any keys that are missing from the data dictionary must have been
        # serialized as None. We need to add them back in.
        for key in type_hints:
            if key not in input_dict:
                kwargs_dict[key] = None

        # Instantiate the object.
        new_obj = cls(**kwargs_dict)

        # Call the after_deserialize method immediately after instantiation.
        new_obj.after_deserialize()

        # Return the object, now that we've called the after_deserialize method.
        return new_obj

    @classmethod
    def _get_attr_names_to_pass_to_init(cls) -> list[str]:
        """
        Returns a list of attribute names that should be passed to the __init__
        method. This is a simple alias to the builtin get_type_hints function.
        """
        # Return a list of the keys in the type hints for the __init__ method.
        return list(get_input_type_hints(cls.__init__))

    @classmethod
    def _deserialize_value(cls, value: Any, type_hint: Any) -> Any:
        """
        Deserializes a value based on the type hint provided.
        """
        # Check to see if the type_hint is Optional. This is a special case in
        # this module, because we never serialize None. That means, if execution
        # reaches here and the type_hint is Optional, we know that the value
        # cannot be None and must be the actual value.
        if is_optional(type_hint):
            return cls._deserialize_value(value, get_args(type_hint)[0])

        # Check to see if the type hint is a type, or a generic alias. An easy
        # way to do this is to get the origin of the type hint, and check
        # it's None.
        origin = get_origin(type_hint)

        if origin is None:
            # If the origin is None, we have a simple, non-generic type hint.
            # If the type hint is a serializable type, we need to call the
            # from_dict method on it.
            if issubclass(type_hint, Serializable):
                return type_hint.from_dict(value)

            # If the type hint is a numpy array, we need to make sure that we
            # reconstruct the numpy array properly.
            if type_hint == np.ndarray:
                return np.array(value)

            # If execution reaches here, we can just return the value.
            return value

        # Check to see if the type hint is a literal. If it is, we need to check
        # if the value is one of the literals.
        if origin is Literal:
            if value not in get_args(type_hint):
                raise ValueError(
                    f"Value {value} is not in the list of literals: "
                    f"{get_args(type_hint)}"
                )
            return value

        if origin is Union:
            # Get the type hints for the items in the union.
            item_type_hints = get_args(type_hint)

            # Try to deserialize the value with each of the type hints.
            for item_type_hint in item_type_hints:
                try:
                    return cls._deserialize_value(value, item_type_hint)
                except (TypeError, ValueError, AttributeError):
                    continue

            # If we reach here, we've tried all the type hints and none of them
            # worked. Raise an error.
            raise TypeError(f"Failed to deserialize value: {value}")

        # If the type hint is a tuple, we need to deserialize each item in the
        # tuple. Because tuples are sequences, we need to check for them before
        # we check for sequences.
        if issubclass(origin, tuple):
            # Get the type hints for the items in the tuple.
            item_type_hints = get_args(type_hint)

            # There are two options here. The first option is that the tuple
            # is acting as a fixed length sequence, in which case the second
            # argument to get_args will be an ellipsis. Deal with that first.
            if isinstance(item_type_hints[-1], EllipsisType):
                # Get the type hint for the items in the tuple.
                item_type_hint = item_type_hints[0]

                # Deserialize each item in the tuple.
                return tuple(
                    cls._deserialize_value(item, item_type_hint)
                    for item in value
                )

            # The second option is that every item in the tuple has been
            # individually type hinted. In this case, we need to deserialize
            # each item with the corresponding type hint.
            return tuple(
                cls._deserialize_value(item, item_type_hint)
                for item, item_type_hint in zip(value, item_type_hints)
            )

        # Check to see if the type hint is a sequence. If it is, we need to
        # deserialize each item in the sequence.
        if issubclass(origin, Sequence):
            # Get the type hint for the items in the sequence.
            item_type_hint = get_args(type_hint)[0]

            # Deserialize each item in the sequence.
            return [
                cls._deserialize_value(item, item_type_hint) for item in value
            ]

        # If the type hint is a dictionary, we need to deserialize each value in
        # the dictionary.
        if issubclass(origin, Mapping):
            # Get the type hints for the keys and values in the dictionary.
            key_type_hint = get_args(type_hint)[0]
            value_type_hint = get_args(type_hint)[1]

            # Deserialize each value in the dictionary.
            return {
                cls._deserialize_value(
                    key, key_type_hint
                ): cls._deserialize_value(val, value_type_hint)
                for key, val in value.items()
            }

        # If the type hint is a set, we need to deserialize each item in the
        # set.
        if issubclass(origin, set):
            # Get the type hint for the items in the set.
            item_type_hint = get_args(type_hint)[0]

            # Deserialize each item in the set.
            return {
                cls._deserialize_value(item, item_type_hint) for item in value
            }

        # If the type hint is a union, we need to try to deserialize the value
        # with each of the type hints in the union, starting from the first one.
        if issubclass(origin, UnionType):
            # Get the type hints for the items in the union.
            item_type_hints = get_args(type_hint)

            # Try to deserialize the value with each of the type hints.
            for item_type_hint in item_type_hints:
                try:
                    return cls._deserialize_value(value, item_type_hint)
                except (TypeError, ValueError, AttributeError):
                    continue

        # If execution reaches here, we have a type hint that we don't know how
        # to deserialize. Raise an error.
        raise TypeError(f"Failed to deserialize type hint: {type_hint}")

    @staticmethod
    def _import_class(class_name: str) -> Any:
        """
        Dynamically imports a class based on its fully qualified name.
        """
        # Split the class name into its module and class name.
        module_name, class_name = class_name.rsplit(".", 1)

        # Import the module.
        module = import_module(module_name)

        # Get the class from the module.
        return getattr(module, class_name)

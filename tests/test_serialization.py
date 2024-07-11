"""
This script contains tests for the AutoSerialize class.
"""

# We want to test some important protected methods.
# pylint: disable=protected-access

from typing import Literal, Sequence

import numpy as np
import pytest

from auto_serialize import AutoSerialize


class SimpleClass1(AutoSerialize):
    """
    This should be easy to serialize.
    """

    def __init__(self, value: int):
        # The value that we want to put back through __init__ to properly
        # serialize/deserialize. This is the true class data.
        self.value = value


class SimpleClass2(AutoSerialize):
    """
    This should be easy to serialize.
    """

    def __init__(self, value: int):
        # The value that we want to put back through __init__ to properly
        # serialize/deserialize. This is the true class data.
        self._value = value

    @property
    def value(self) -> int:
        """
        Doesn't return the value!
        """
        return 2 * self._value


class SimpleClass3(AutoSerialize):
    """
    This class makes sure that we can properly handle collections.
    """

    def __init__(self, value: list[int]):
        # The value that we want to put back through __init__ to properly
        # serialize/deserialize. This is the true class data.
        self.value = value


class SimpleClass4(AutoSerialize):
    """
    This class makes sure that we raise an error when we can't serialize. We
    should be able to deserialize though, if given a dictionary with a key
    called "value" (which we can figure out from the __init__ method's type
    hints).
    """

    def __init__(self, value: int):
        # The value that we want to put back through __init__ to properly
        # serialize/deserialize. This is the true class data. We will never be
        # able to find this attribute's name, because it's unrelated to the
        # name that it has when it passes through __init__.
        self.who_dis = value


class SimpleClass5(AutoSerialize):
    """
    This class makes sure that we serialize None to nothing at all.
    """

    def __init__(self, value: str | None) -> None:
        self.value = value


class SimpleClass6(AutoSerialize):
    """
    This class makes sure that we can serialize lists of AutoSerialize
    objects.
    """

    def __init__(self, values: list[SimpleClass1]) -> None:
        self.values = values


class SimpleClass7(AutoSerialize):
    """
    This class makes sure that we can serialize/deserialize dictionaries where
    the values are AutoSerialize objects.
    """

    def __init__(self, values: dict[str, SimpleClass1]) -> None:
        self.values = values


class SimpleClass8(AutoSerialize):
    """
    This class makes sure that we can handle unions. The deserializer should be
    clever enough to figure out which option is the right one.
    """

    def __init__(self, value: int | str) -> None:
        self.value = value


class SimpleClass9(AutoSerialize):
    """
    This class makes sure that we can serialize/deserialize literals.
    """

    def __init__(self, value: Literal["test_literal"] | None) -> None:
        self.value = value


class SimpleClass10(AutoSerialize):
    """
    Make sure we can serialize/deserialize numpy arrays.
    """

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class NotSoSimpleClass1(AutoSerialize):
    """
    This class makes sure that we can serialize/deserialize heavily nested
    generic types, where a very deeply nested type is an AutoSerialize object.
    """

    def __init__(
        self, values: list[list[dict[str, list[SimpleClass1]]]]
    ) -> None:
        self.values = values


class NotSoSimpleClass2(AutoSerialize):
    """
    This class makes sure that we can serialize/deserialize dictionaries where
    the values are lists of AutoSerialize objects.
    """

    def __init__(
        self, values: dict[str, list[dict[str, list[SimpleClass1]]]]
    ) -> None:
        self._values = values


class MyClass(AutoSerialize):
    """
    A class used in the demo.
    """

    def __init__(self, a: int, b: tuple[dict[str, list[float]], str]):
        self._a = a
        self.b = b


class OtherDemoClass(AutoSerialize):
    """
    A second class that we use in the demo.
    """

    def __init__(self, value: int) -> None:
        self._value = value


class DemoClass(AutoSerialize):
    """
    This class demos the use of the AutoSerialize class.
    """

    def __init__(
        self,
        a: int,
        b: list[float] | float,
        c: dict[str, OtherDemoClass | list[int]],
        arr: np.ndarray,
        maybe: str | None = None,
    ) -> None:
        self._a = a
        self._b = b
        self.c = c
        self.arr = arr
        self._maybe = maybe

    def before_serialize(self) -> None:
        self._a += 1

    def after_deserialize(self) -> None:
        if self._maybe is None:
            self._maybe = "world!"


class ChildClass1(SimpleClass1):
    """
    This class is a child of SimpleClass1.
    """

    def __init__(self, value: int, value_2: str) -> None:
        super().__init__(value)

        self.value_2 = value_2


class ListClass1(AutoSerialize):
    """
    This class has a list of SimpleClass1 objects. By polymorphism, it should
    be able to also take lists of ChildClass1 objects.
    """

    def __init__(self, values: Sequence[SimpleClass1]) -> None:
        self.values = values


class SerializationCallbackClass(AutoSerialize):
    """
    This class has a custom serialization callback.
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def before_serialize(self) -> None:
        self.value += 1


class DeserializationCallbackClass(AutoSerialize):
    """
    This class has a custom deserialization callback.
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def after_deserialize(self) -> None:
        self.value -= 1


class EllipsisTypeHintClass(AutoSerialize):
    """
    A simple class with ellipsis type hints in a tuple.
    """

    def __init__(self, value: tuple[int, ...]) -> None:
        self.value = value


class TupleNoEllipsisTypeHint(AutoSerialize):
    """
    A simple class with a tuple type hint without an ellipsis.
    """

    def __init__(
        self,
        value: tuple[int, Literal["hello"], Literal["world"], SimpleClass1],
    ):
        self.value = value


def test_simple_class1_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_value = 5
    simple = SimpleClass1(original_value)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple2 = SimpleClass1.from_dict(data)

    # Check that the value is the same.
    assert simple2.value == original_value

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple2 = SimpleClass1.from_json_bytes(data)
    assert simple2.value == original_value

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple2 = SimpleClass1.from_yaml(data)
    assert simple2.value == original_value


def test_simple_class_2_attr_names_to_serialize():
    """
    Test the _get_attr_names_to_serialize method of SimpleClass2.
    """
    # Create an instance of the class.
    simple = SimpleClass2(5)

    # Get the attribute names that should be serialized.
    attr_names = simple._get_attr_names_to_serialize()

    # Check that the attribute names are correct.
    assert attr_names == ["_value"]


def test_simple_class2_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_value = 5
    simple = SimpleClass2(original_value)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple2 = SimpleClass2.from_dict(data)

    # Check that the value is the same.
    assert simple2._value == original_value

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple2 = SimpleClass2.from_json_bytes(data)
    assert simple2._value == original_value

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple2 = SimpleClass2.from_yaml(data)
    assert simple2._value == original_value


def test_simple_class3_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_value = [5, 6, 7]
    simple = SimpleClass3(original_value)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass3.from_dict(data)

    # Check that the value is the same.
    assert simple.value == original_value

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass3.from_json_bytes(data)
    assert simple.value == original_value

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass3.from_yaml(data)
    assert simple.value == original_value


def test_simple_class4_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_value = 5
    simple = SimpleClass4(original_value)

    # Serialization should throw.
    with pytest.raises(AttributeError):
        # Serialize the instance to a dictionary.
        data = simple.to_dict()

    # Deserialization should work though, if we pass in a dictionary with an
    # appropriate key.
    data = {
        "value": original_value,
        "class_name": "test_serialization.SimpleClass4",
    }
    simple = SimpleClass4.from_dict(data)

    # Check that the value is the same.
    assert simple.who_dis == original_value

    # Now do the same round trip to json.


def test_simple_class5_none_empty():
    """
    This test uses the fact that SimpleClass5 has a value that can be None to
    verify that we don't serialize None to anything.
    """
    # Create an instance of the class.
    simple = SimpleClass5(None)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Check that the value is not in the dictionary.
    assert "value" not in data
    assert not data

    # Deserialize the empty dictionary back into an instance of the class.
    simple = SimpleClass5.from_dict({})
    assert simple.value is None

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass5.from_json_bytes(data)
    assert simple.value is None

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass5.from_yaml(data)
    assert simple.value is None


def test_simple_class5_none_present():
    """
    This test uses the fact that SimpleClass5 has a value that can be None to
    verify that we don't serialize None to anything.
    """
    # Create an instance of the class.
    simple = SimpleClass5("hello")

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Check that the value is in the dictionary.
    assert "value" in data
    assert data["value"] == "hello"

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass5.from_dict(data)

    # Check that the value is the same.
    assert simple.value == "hello"

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass5.from_json_bytes(data)
    assert simple.value == "hello"

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass5.from_yaml(data)
    assert simple.value == "hello"


def test_simple_class6_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_values = [SimpleClass1(5), SimpleClass1(6), SimpleClass1(7)]
    original_simple = SimpleClass6(original_values)

    # Serialize the instance to a dictionary.
    data = original_simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass6.from_dict(data)

    # Check that the value is the same.
    assert [x.value for x in simple.values] == [
        x.value for x in original_simple.values
    ]

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass6.from_json_bytes(data)
    assert [x.value for x in simple.values] == [
        x.value for x in original_simple.values
    ]

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass6.from_yaml(data)
    assert [x.value for x in simple.values] == [
        x.value for x in original_simple.values
    ]


def test_simple_class6_fully_serialized():
    """
    It's possible for the above test to pass if we're not completely serializing
    all the SimpleClass1 instances all the way down. This test makes sure that
    we are.
    """
    # Create an instance of the class.
    original_values = [SimpleClass1(5), SimpleClass1(6), SimpleClass1(7)]
    original_simple = SimpleClass6(original_values)

    # Serialize the instance to a dictionary.
    data = original_simple.to_dict()

    # Check that the value is the same.
    assert data == {
        "values": [
            {"value": 5},
            {"value": 6},
            {"value": 7},
        ]
    }


def test_simple_class7_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_values = {
        "hello": SimpleClass1(5),
        "world": SimpleClass1(6),
        "foo": SimpleClass1(7),
    }
    original_simple = SimpleClass7(original_values)

    # Serialize the instance to a dictionary.
    data = original_simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass7.from_dict(data)

    # Check that the value is the same.
    assert {key: value.value for key, value in simple.values.items()} == {
        key: value.value for key, value in original_simple.values.items()
    }

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass7.from_json_bytes(data)
    assert {key: value.value for key, value in simple.values.items()} == {
        key: value.value for key, value in original_simple.values.items()
    }

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass7.from_yaml(data)
    assert {key: value.value for key, value in simple.values.items()} == {
        key: value.value for key, value in original_simple.values.items()
    }


def test_simple_class8_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_value = 5
    simple = SimpleClass8(original_value)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass8.from_dict(data)

    # Check that the value is the same.
    assert simple.value == original_value

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass8.from_json_bytes(data)
    assert simple.value == original_value

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass8.from_yaml(data)
    assert simple.value == original_value


def test_simple_class8_round_trip_str():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_value = "hello"
    simple = SimpleClass8(original_value)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass8.from_dict(data)

    # Check that the value is the same.
    assert simple.value == original_value

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass8.from_json_bytes(data)
    assert simple.value == original_value

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass8.from_yaml(data)
    assert simple.value == original_value


def test_simple_class9_correct_literal():
    """
    Make sure that we can serialize/deserialize the correct literal when our
    type hint is a Literal.
    """
    # Create an instance of the class.
    simple = SimpleClass9("test_literal")

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass9.from_dict(data)

    # Check that the value is the same.
    assert simple.value == "test_literal"

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = SimpleClass9.from_json_bytes(data)
    assert simple.value == "test_literal"

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass9.from_yaml(data)
    assert simple.value == "test_literal"


def test_simple_class9_incorrect_literal():
    """
    Make sure that we raise an error when we try to serialize/deserialize an
    incorrect literal.
    """
    # Create an instance of the class.
    simple = SimpleClass9("not_test_literal")  # type: ignore

    # Serialization should not throw - it's on the user if they misconstruct
    # their own types. It isn't really our job to retroactively raise
    # an exception, when the damage has already been done at serialization time.
    data = simple.to_dict()

    # Deserialization should throw, though. We should refuse to construct
    # invalid types.
    with pytest.raises(ValueError):
        # Deserialize the dictionary back into an instance of the class.
        simple = SimpleClass9.from_dict(data)


def test_simple_class10_round_trip():
    """
    Make sure we can serialize and deserialize numpy arrays. Here we do an
    additional serialization all the way to a byte array, just because json
    doesn't natively support numpy arrays, so this test is a bit more thorough.
    """
    # Create an instance of the class.
    original_value = np.array([1, 2, 3])
    simple = SimpleClass10(original_value)

    # Serialize the instance to a dictionary.
    data = simple.to_json_bytes()

    # Deserialize the dictionary back into an instance of the class.
    simple = SimpleClass10.from_json_bytes(data)

    # Check that the value is the same.
    assert isinstance(simple.value, np.ndarray)
    assert np.array_equal(simple.value, original_value)

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = SimpleClass10.from_yaml(data)
    assert isinstance(simple.value, np.ndarray)


def test_not_so_simple_class1_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_values = [
        [{"hello": [SimpleClass1(5), SimpleClass1(6), SimpleClass1(7)]}]
    ]
    original_simple = NotSoSimpleClass1(original_values)

    # Serialize the instance to a dictionary.
    data = original_simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = NotSoSimpleClass1.from_dict(data)

    # Check that the value is the same.
    assert simple.values == original_simple.values

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = NotSoSimpleClass1.from_json_bytes(data)
    assert simple.values == original_simple.values

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = NotSoSimpleClass1.from_yaml(data)
    assert simple.values == original_simple.values


def test_not_so_simple_class2_round_trip():
    """
    Test the serialization of a simple class.
    """
    # Create an instance of the class.
    original_values = {
        "hello": [
            {"world": [SimpleClass1(5), SimpleClass1(6), SimpleClass1(7)]}
        ]
    }
    original_simple = NotSoSimpleClass2(original_values)

    # Serialize the instance to a dictionary.
    data = original_simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple = NotSoSimpleClass2.from_dict(data)

    # Check that the value is the same.
    assert simple._values == original_simple._values

    # Now do the same round trip to json.
    data = simple.to_json_bytes()
    simple = NotSoSimpleClass2.from_json_bytes(data)
    assert simple._values == original_simple._values

    # Now do the same round trip to yaml.
    data = simple.to_yaml()
    simple = NotSoSimpleClass2.from_yaml(data)
    assert simple._values == original_simple._values


def test_equal():
    """
    Test the equality operator. We should be able to compare two instances of
    the same class and get True if they have the same data, for any class
    that inherits from AutoSerialize.
    """
    # Create an instance of the class.
    simple1 = SimpleClass7({"hello": SimpleClass1(5)})
    simple2 = SimpleClass7({"hello": SimpleClass1(5)})
    simple3 = SimpleClass7({"hello": SimpleClass1(6)})  # Different value.
    simple4 = SimpleClass7({"world": SimpleClass1(5)})  # Different key.

    # Check that the instances are equal.
    assert simple1 == simple2
    assert simple1 != simple3
    assert simple1 != simple4
    assert simple2 != simple3
    assert simple2 != simple4
    assert simple3 != simple4


def test_hash():
    """
    Test the hash method. We should be able to hash two instances of the same
    class and get the same hash if they have the same data, for any class that
    inherits from AutoSerialize.
    """
    simple1 = SimpleClass7({"hello": SimpleClass1(5)})
    simple2 = SimpleClass7({"hello": SimpleClass1(5)})
    simple3 = SimpleClass7({"hello": SimpleClass1(6)})  # Different value.
    simple4 = SimpleClass7({"world": SimpleClass1(5)})  # Different key.

    assert hash(simple1) == hash(simple2)
    assert hash(simple1) != hash(simple3)
    assert hash(simple1) != hash(simple4)
    assert hash(simple2) != hash(simple3)
    assert hash(simple2) != hash(simple4)
    assert hash(simple3) != hash(simple4)


def test_child_class1_round_trip():
    """
    Test the serialization of a child class.
    """
    # Create an instance of the class.
    original_value = 5
    original_value_2 = "hello"
    simple = ChildClass1(original_value, original_value_2)

    # Serialize the instance to a dictionary.
    data = simple.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    simple2 = ChildClass1.from_dict(data)

    # Check that the value is the same.
    assert simple2.value == original_value
    assert simple2 == simple


def test_list_of_child_class1_round_trip():
    """
    Test the serialization of a list of child classes.
    """
    # Create an instance of the class.
    original_values = [ChildClass1(5, "hello"), ChildClass1(6, "world")]
    obj1 = ListClass1(original_values)

    # Serialize the instance to a dictionary.
    data = obj1.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    obj2 = ListClass1.from_dict(data)

    # Check that the value is the same.
    assert [x.value for x in obj2.values] == [x.value for x in obj1.values]
    assert obj1 == obj2

    # Now do the same round trip to json.
    data = obj1.to_json_bytes()
    obj2 = ListClass1.from_json_bytes(data)
    assert [x.value for x in obj2.values] == [x.value for x in obj1.values]
    assert obj1 == obj2

    # Now do the same round trip to yaml.
    data = obj1.to_yaml()
    obj2 = ListClass1.from_yaml(data)
    assert [x.value for x in obj2.values] == [x.value for x in obj1.values]
    assert obj1 == obj2


def test_list_different_types_round_trip():
    """
    This test is really tricky to make work. We accept a list of anything that
    satisfies the type hint, which could be a collection of different instances
    of subclasses of the same base class. To deserialize them, we need to
    dynamically work out which subclass to use.

    To make it extra hard, we serialize all the way to json and back.
    """
    original_values = [SimpleClass1(5), ChildClass1(6, "hello")]
    obj1 = ListClass1(original_values)

    # Serialize the instance to a dictionary.
    data = obj1.to_json_bytes()

    # Deserialize the dictionary back into an instance of the class.
    obj2 = ListClass1.from_json_bytes(data)

    # Check that the value is the same.
    assert obj1 == obj2

    # Make sure that the data json string is exactly what we expect. The key
    # here is that we need exactly one class name - only for the ChildClass, as
    # we can infer the SimpleClass from the type hint.
    assert data == (
        b'{"values":[{"value":5},{"value":6,"value_2":"hello","class_name":'
        b'"test_serialization.ChildClass1"}]}'
    )

    # Now do the same round trip to yaml.
    data = obj1.to_yaml()
    obj2 = ListClass1.from_yaml(data)
    assert obj1 == obj2


def test_serialization_callback():
    """
    Test the serialization callback.
    """
    # Create an instance of the class.
    input_value = 4
    test = SerializationCallbackClass(input_value)

    # Serialize the instance to a dictionary.
    data = test.to_dict()

    # Make sure that, in the dictionary, the value has been incremented.
    assert data["value"] == input_value + 1

    # Now deserialize the dictionary back into an instance of the class.
    test2 = SerializationCallbackClass.from_dict(data)

    # Check that the value is still incremented.
    assert test2.value == input_value + 1

    # Now do the same round trip to json.
    test = SerializationCallbackClass(input_value)
    data = test.to_json_bytes()
    test2 = SerializationCallbackClass.from_json_bytes(data)
    assert test2.value == input_value + 1

    # Now do the same round trip to yaml.
    test = SerializationCallbackClass(input_value)
    data = test.to_yaml()
    test2 = SerializationCallbackClass.from_yaml(data)
    assert test2.value == input_value + 1


def test_deserialization_callback():
    """
    Test the deserialization callback.
    """
    # Create an instance of the class.
    input_value = 4
    test = DeserializationCallbackClass(input_value)

    # Serialize the instance to a dictionary.
    data = test.to_dict()

    # Make sure that the value in the dictionary hasn't changed.
    assert data["value"] == input_value

    # Now deserialize the dictionary back into an instance of the class.
    test2 = DeserializationCallbackClass.from_dict(data)

    # Check that the value has been decremented.
    assert test2.value == input_value - 1

    # Now do the same round trip to json.
    data = test.to_json_bytes()
    test2 = DeserializationCallbackClass.from_json_bytes(data)
    assert test2.value == input_value - 1

    # Now do the same round trip to yaml.
    data = test.to_yaml()
    test2 = DeserializationCallbackClass.from_yaml(data)
    assert test2.value == input_value - 1


def test_ellipsis_tuple():
    """
    Test that we can serialize/deserialize a tuple with an ellipsis.
    """
    # Create an instance of the class.
    test = EllipsisTypeHintClass((1, 2, 3))

    # Serialize the instance to a dictionary.
    data = test.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    test2 = EllipsisTypeHintClass.from_dict(data)

    # Check that the value is the same.
    assert test2.value == (1, 2, 3)

    # Now do the same round trip to json.
    data = test.to_json_bytes()
    test2 = EllipsisTypeHintClass.from_json_bytes(data)
    assert test2.value == (1, 2, 3)

    # Now do the same round trip to yaml.
    data = test.to_yaml()
    test2 = EllipsisTypeHintClass.from_yaml(data)
    assert test2.value == (1, 2, 3)


def test_explicit_tuple():
    """
    Make sure that we can serialize/deserialize a tuple where every element has
    a different type with specific deserialization needs.
    """
    simple_1 = SimpleClass1(5)
    test = TupleNoEllipsisTypeHint((1, "hello", "world", simple_1))

    # Serialize the instance to a dictionary.
    data = test.to_dict()

    # Deserialize the dictionary back into an instance of the class.
    test2 = TupleNoEllipsisTypeHint.from_dict(data)

    # Check that the value is the same.
    assert test2.value == (1, "hello", "world", simple_1)

    # Now do the same round trip to json.
    data = test.to_json_bytes()
    test2 = TupleNoEllipsisTypeHint.from_json_bytes(data)
    assert test2.value == (1, "hello", "world", simple_1)

    # Now do the same round trip to yaml.
    data = test.to_yaml()
    test2 = TupleNoEllipsisTypeHint.from_yaml(data)
    assert test2.value == (1, "hello", "world", simple_1)


def test_first_demo_class():
    """
    Make sure that the simple demo works.
    """
    test = MyClass(2, ({"hello": [4, 2]}, "world"))

    json_str = test.to_json()

    test_again = MyClass.from_json(json_str)

    assert test_again == test  # True.


def test_demo_class():
    """
    Make sure that the demo in the readme works.
    """
    other_demo_class = OtherDemoClass(5)
    demo = DemoClass(
        1, 3.141, {"hello, ": other_demo_class}, np.array([4, 5, 6])
    )

    json_str = demo.to_json()
    demo_again = DemoClass.from_json(json_str)

    assert demo_again._a == 2
    assert demo_again._b == 3.141
    assert demo_again.c == {"hello, ": other_demo_class}
    assert np.array_equal(demo_again.arr, np.array([4, 5, 6]))
    assert isinstance(demo_again.arr, np.ndarray)
    assert demo_again._maybe == "world!"

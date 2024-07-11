# auto_serialize

This is a simple two-file serialization/deserialization library.

This is not a data validation library. For data validation, use pydantic.
If you have a polygot codebase, use pydantic.
You won't care about the schema, but your colleagues will.

## Installation

Install with `pip install auto_serialize`

## Usage

Just inherit from `AutoSerialize`!

```python
from auto_serialize import AutoSerialize

class MyClass(AutoSerialize):
    def __init__(self, a: int, b: tuple[dict[str, list[float]], str]):
        self._a = a
        self.b = b

test = MyClass(2, ({"hello": [4, 2]}, "world"))

# Go to and from json.
json_str = test.to_json()
test_again = MyClass.from_json(json_str)

json_str == '{"a":2,"b":[{"hello":[4,2]},"world"]}' # True
test_again == test # True.

# We can also serialize to yaml. More formats coming soon :)
test.to_yaml() == 'a: 2\nb: !!python/tuple\n- hello:\n  - 4\n  - 2\n- world\n' # True
```

`AutoSerialize` doesn't just support inbuilt types. It also supports unions,
numpy arrays, heavily nested data types, and serialization/deserialization
callbacks.

```python
class OtherDemoClass(AutoSerialize):
    def __init__(self, value: int) -> None:
        self._value = value


class DemoClass(AutoSerialize):
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

other_demo_class = OtherDemoClass(5)
demo = DemoClass(
    1, 3.141, {"hello, ": other_demo_class}, np.array([4, 5, 6])
)

json_str = demo.to_json()
demo_again = DemoClass.from_json(json_str)

# The following assertions all pass.
assert demo_again._a == 2
assert demo_again._b == 3.141
assert demo_again.c == {"hello, ": other_demo_class}
assert np.array_equal(demo_again.arr, np.array([4, 5, 6]))
assert isinstance(demo_again.arr, np.ndarray)
assert demo_again._maybe == "world!"
```

Some other handy things you get from using AutoSerialize:

- A `deepcopy()` method
  - Currently implemented by serializing to json, then deserializing back.
- `__eq__`, `__neq__` and `__hash__` default implementations.
  - Currently implemented by mapping your AutoSerialize objects to python
    dictionaries.

## Why?

This is a self serving repository.

I need to serialize a lot of stuff, and I have had this need for a long time.
I have no time for 3rd party libraries that don't support my data format.
If I need to serialize a numpy array with NaNs, I need to be able to implement
that functionality in a very small amount of time.

I have posted this online in the hope that the internet will bully me into
supporting more data formats, to build something really meaningful.

## Implementation details; gotchas

`AutoSerialize` works by inspecting the type hints on `__init__`. If you don't
use type hints, it won't work.

Union resolution is optimistic, going from left to right, and not all generics
are checked for. As a result, if the above DemoClass was type hinted
`c: dict[str, list[int] | OtherDemoClass],`
The OtherDemoClass would actually get deserialized as a list! This is because
the int generic isn't checked for, and OtherDemoClass is serialized to a dict,
which is iterable! :scream:

But, because a list of integers could never be deserialized as OtherDemoClass,
using the ordering
`c: dict[str, OtherDemoClass | list[int]],`
will always work. If c was a list of integers, `AutoSerialize` would start by
trying to deserialize the list of integers as an instance of OtherDemoClass.
This would fail, and then `AutoSerialize` would fallback to the list[int]
option, which would work! :tada:

## Contributing

If you have an example class that won't serialize, submit an issue. I'll fix it
and add your class to the test suite! Of course, feel free to submit a PR if
you're up to fixing it yourself. :muscle:

## Roadmap

#### Formats

I have an interest in hdf5 and protobuf formats, so it's likely that they will
both end up being supported in the near future.
If you have a particular need for either of these formats, please submit an
issue. That will be enough motivation for me to get it done in no time.

#### Schema

It would be pretty easy to make this library have the same validation
functionality that pydantic has. I haven't pushed myself to implement this
because pydantic already exists. If there was ever a need, it would be done.

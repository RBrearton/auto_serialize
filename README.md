# auto_serialize

This is a simple two-file serialization/deserialization library.
This is not a data validation library. For data validation, use pydantic.
python
### Installation

Install with `pip install auto_serialize`

### Usage

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

test_again == test # True.
```

AutoSerialize doesn't just support inbuilt types. It also supports unions, numpy
arrays, heavily nested data types, and serialization/deserialization callbacks.

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


### Implementation details; gotchas

Union resolution is optimistic, going from left to right, and not all generics
are checked for. As a result, if the above DemoClass was type hinted
`c: dict[str, list[int] | OtherDemoClass],`
The OtherDemoClass would actually get deserialized as a list! This is because
the int generic isn't checked for, and OtherDemoClass is serialized to a dict,
which is iterable! :scream:

But, because a list of integers could never be deserialized as OtherDemoClass,
using the ordering 
`c: dict[str, OtherDemoClass | list[int]],`
will always work. If c was a list of integers, AutoSerialize would start by
trying to deserialize the list of integers as an instance of OtherDemoClass.
This would fail, and then AutoSerialize would fallback to the list[int] option,
which would work! :tada:
# auto_serialize

This is an extremely simple, 2 file library, intended to "just work" when
serializing typical python data structures. There is no focus on data
validation - pydantic is already an outstanding tool for this purpose.

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

json_str = test.to_json()

test_again = MyClass.from_json(test)

assert test_again == test # True.
```

AutoSerialize supports unions, numpy arrays, and heavily nested data types. It
works by looking at the type hints on your __init__ method, and searching for
attributes with names similar to these 
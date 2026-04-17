class ShapeKeys:
    def __init__(self, value: float):
        self._value = value

    def __str__(self):
        return f"Value: {str(self._value)}"

    def to_dict(self):
        return {"shape_keys": {"value": f"{self._value:.2f}"}}

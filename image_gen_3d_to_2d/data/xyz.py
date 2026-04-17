class XYZ:
    def __init__(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z

    def __str__(self):
        return f"{self._x:.2f},{self._y:.2f},{self._z:.2f}"

    def to_dict(self):
        return {"x": f"{self._x:.2f}", "y": f"{self._y:.2f}", "z": f"{self._z:.2f}"}

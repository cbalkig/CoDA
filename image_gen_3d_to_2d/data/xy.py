class XY:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def __str__(self):
        return f"{self._x:.2f},{self._y:.2f}"

    def to_dict(self):
        return {"x": f"{self._x:.2f}", "y": f"{self._y:.2f}"}

from typing import Optional


class HSV:
    def __init__(self, hue: float, saturation: float, value: float):
        self._hue = hue
        self._saturation = saturation
        self._value = value

    def __str__(self):
        return f"{self._hue:.2f},{self._saturation:.2f},{self._value:.2f}"

    def to_dict(self):
        return {"hue": self._hue, "saturation": self._saturation, "value": self._value}


class UV:
    def __init__(self, scale: float, rotation: float, location: float):
        self._scale = scale
        self._rotation = rotation
        self._location = location

    def __str__(self):
        return f"{self._scale:.2f},{self._rotation:.2f},{self._location:.2f}"

    def to_dict(self):
        return {"scale": self._scale, "rotation": self._rotation, "location": self._location}


class Materials:
    def __init__(self, hsv: HSV, specular: float, roughness: float, metallic: Optional[float] = None,
                 jitter_hsv: Optional[HSV] = None,
                 uv: Optional[UV] = None):
        self._hsv = hsv
        self._specular = specular if specular is not None else -1
        self._roughness = roughness
        self._metallic = metallic if metallic is not None else -1
        self._jitter_hsv = jitter_hsv
        self._uv = uv

    def __str__(self):
        return f"HSV: {str(self._hsv)}, Specular: {str(self._specular):.2f}, Roughness: {str(self._roughness):.2f}, Metallic: {str(self._metallic):.2f}, Jitter HSV: {str(self._jitter_hsv)}, UV: {str(self._uv)}"

    def to_dict(self):
        return {
            "materials": {"hsv": self._hsv.to_dict(), "specular": f"{self._specular:.2f}",
                          "roughness": f"{self._roughness:.2f}",
                          "metallic": f"{self._metallic:.2f}",
                          "jitter_hsv": self._jitter_hsv.to_dict() if self._jitter_hsv else None,
                          "uv": self._uv.to_dict() if self._uv else None}}

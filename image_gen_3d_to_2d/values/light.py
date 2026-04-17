from typing import List, Dict

from data.xyz import XYZ


class LightKeys:
    def __init__(self, light_type: str, energy: float, temperature: float, distance_to_objects: Dict[str, float],
                 location: XYZ):
        self._light_type = light_type
        self._energy = energy
        self._temperature = temperature
        self._distance_to_objects = distance_to_objects
        self._location = location

    def __str__(self):
        return f"Light Type: {self._light_type}, Energy: {self._energy:.2f}, Temperature: {self._temperature:.2f}, Distance to Objects:{str(self._distance_to_objects)}, Location: {str(self._location)}"

    def to_dict(self):
        return {"light_type": self._light_type, "energy": self._energy, "temperature": self._temperature,
                "distance_to_objects": self._distance_to_objects, "location": self._location.to_dict()}


class LightData:
    def __init__(self, hdr_name: str, rot_deg: float, exposure: float, light_keys: List[LightKeys]):
        self._hdr_name = hdr_name
        self._rot_deg = rot_deg
        self._exposure = exposure
        self._light_keys = light_keys

    def __str__(self):
        keys_repr = ", ".join(str(k) for k in self._light_keys)
        return (
            f"HDR Name: {self._hdr_name}, "
            f"Rotation Degree: {self._rot_deg:.2f}, "
            f"Exposure: {self._exposure:.2f}, "
            f"Light Keys: [{keys_repr}]"
        )

    def to_dict(self):
        return {
            "light": {
                "hdr_name": self._hdr_name,
                "rot_deg": f"{self._rot_deg:.2f}",
                "exposure": f"{self._exposure:.2f}",
                "light_keys": [k.to_dict() for k in self._light_keys],
            }
        }

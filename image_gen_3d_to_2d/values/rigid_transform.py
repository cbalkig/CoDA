from data.xyz import XYZ


class RigidTransformData:
    def __init__(self, translation: XYZ, rotation: XYZ, scale: float):
        self._translation = translation
        self._rotation = rotation
        self._scale = scale

    def __str__(self):
        return f"Translation: {str(self._translation)}, Rotation: {str(self._rotation)}, Scale: {str(self._scale)}"

    def to_dict(self):
        return {"rigid_transform": {"translation": self._translation.to_dict(), "rotation": self._rotation.to_dict(),
                                    "scale": f"{self._scale:.2f}"}}

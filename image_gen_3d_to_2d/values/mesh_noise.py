class MeshNoise:
    def __init__(self, strength: float):
        self._strength = strength

    def __str__(self):
        return f"Strength: {str(self._strength)}"

    def to_dict(self):
        return {"mesh_noise": {"strength": f"{self._strength:.2f}"}}

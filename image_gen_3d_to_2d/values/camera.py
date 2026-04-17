from data.xyz import XYZ


class CameraData:
    def __init__(self, azimuth: float, elevation: float, radius: float, focal_mm: float, aspect_ratio: float,
                 fstop: float, fit_margin: float, cam_location: XYZ):
        self._azimuth = azimuth
        self._elevation = elevation
        self._radius = radius
        self._focal_mm = focal_mm
        self._aspect_ratio = aspect_ratio
        self._fstop = fstop
        self._fit_margin = fit_margin
        self._cam_location = cam_location

    def __str__(self):
        return f"Azimuth: {self._azimuth:.2f}, Elevation: {self._elevation:.2f}, Radius: {self._radius:.2f}, Focal mm: {self._focal_mm}, Aspect Ratio: {self._aspect_ratio}, FStop: {self._fstop:.2f}, Fit Margin: {self._fit_margin:.2f}, Cam Location: {str(self._cam_location)}"

    def to_dict(self):
        return {
            "camera": {"azimuth": f"{self._azimuth:.2f}", "elevation": f"{self._elevation:.2f}",
                       "radius": f"{self._radius:.2f}", "lens": f"{self._focal_mm:.2f}",
                       "aspect_ratio": f"{self._aspect_ratio:.2f}", "fstop": f"{self._fstop:.2f}",
                       "fit_margin": f"{self._fit_margin:.2f}", "cam_location": self._cam_location.to_dict()}}

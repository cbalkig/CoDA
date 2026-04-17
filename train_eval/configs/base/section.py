class Section:
    def __init__(self, data):
        # normalize keys to strings once
        self._data = {str(k): v for k, v in (data or {}).items()}

    def __iter__(self):
        # lets: for raw_key in cfg:
        return iter(self._data.keys())

    def __getitem__(self, key):
        # all keys stored as str
        return self._data[str(key)]

    def get(self, key, fallback=None):
        return self._data.get(str(key), fallback)

    def getint(self, key, fallback=None):
        v = self.get(key, fallback)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"Expected int for '{key}', got {v!r}")

    def getfloat(self, key, fallback=None):
        v = self.get(key, fallback)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Expected float for '{key}', got {v!r}")

    def getboolean(self, key, fallback=None):
        v = self.get(key, fallback)
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "on"}: return True
            if s in {"0", "false", "no", "off"}: return False
        raise ValueError(f"Expected bool for '{key}', got {v!r}")

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __contains__(self, key):
        return str(key) in self._data

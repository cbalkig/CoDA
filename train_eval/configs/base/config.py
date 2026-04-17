from typing import Dict, Any

from configs.base.section import Section


class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data or {}

    def __getitem__(self, section: str) -> Section:
        return Section(self._data.get(section, {}))

    def get_section_names(self):
        return list(self._data.keys())

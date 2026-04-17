from typing import Optional

from data.types.data_type import DataType
from data.types.domain_type import DomainType


class DataTag:
    def __init__(self, domain: DomainType, data_type: DataType, identifier: Optional[str] = None):
        self.domain = domain
        self.data_type = data_type
        self.identifier = identifier.lower().replace("-", "_").replace(" ", "_") if identifier is not None else None

    def __eq__(self, other):
        if not isinstance(other, DataTag):
            return False
        return self.domain == other.domain and self.data_type == other.data_type and self.identifier == other.identifier

    def __hash__(self):
        return hash((self.domain, self.data_type, str(self.identifier)))

    def __str__(self):
        return f'Domain: {self.domain} - Data Type: {self.data_type} - Identifier: {self.identifier}'

    @property
    def short_tag(self) -> str:
        if self.identifier is None:
            return f'{str(self.domain)}_{str(self.data_type)}'
        else:
            return f'{str(self.domain)}_{self.identifier}_{str(self.data_type)}'

    @property
    def tag(self) -> str:
        if self.identifier is None:
            return f'{str(self.domain)} - {str(self.data_type)}'
        else:
            return f'{str(self.domain)} {self.identifier} - {str(self.data_type)}'

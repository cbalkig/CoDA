from data.data_tag import DataTag
from data.types.domain_type import DomainType


class ModelTag:
    def __init__(self, data_tag: DataTag, eval_on: DomainType):
        self.data_tag = data_tag
        self.eval_on = eval_on

    def __eq__(self, other):
        if not isinstance(other, ModelTag):
            return False
        return self.data_tag == other.data_tag and self.eval_on == other.eval_on

    def __hash__(self):
        return hash((self.data_tag, self.eval_on))

    def __str__(self) -> str:
        return f'Data Tag: {self.data_tag} - Eval on Model: {self.eval_on}'

    @property
    def best_model_tag(self) -> str:
        return f'best_{self.short_tag}'

    @property
    def short_tag(self) -> str:
        return f'{self.data_tag.short_tag}_on_{str(self.eval_on)}'

    @property
    def tag(self) -> str:
        return f'{self.data_tag.tag} - {str(self.eval_on)}'

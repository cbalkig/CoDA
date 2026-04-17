from data.file.path import StoragePath


class ModelConfiguration:
    def __init__(self, folder_path: StoragePath, model_id: str) -> None:
        self.model_id = model_id
        self.path = folder_path.join(self.model_id)

from util.abstract_file_util import AbstractFileUtil


class FileUtil(AbstractFileUtil):
    def __init__(self):
        super().__init__("balki-phd", "credentials.json")

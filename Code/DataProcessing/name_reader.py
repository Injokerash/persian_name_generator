from typing import List


class NameReader:

    def __init__(self, file_name : str) -> None:
        self.file_name = file_name
        self.names : List[str] = []

        with open(self.file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name = self.normalize(line)
                self.names.append(name)

    def normalize(self, name : str) -> str:
        name = name.lower()
        name = name.strip()
        name = name.replace(' ', '')
        name = name.replace('\n', '')
        return name
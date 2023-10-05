from typing import Dict, List
import numpy as np

from Code.DataProcessing.name_reader import NameReader

class NameDataSet:
    def __init__(self, file_name : str, add_padding : int = 0) -> None:
        self.name_reader : NameReader = NameReader(file_name)
        self.padding : int = max(add_padding, 0)
        self.padded_names : List[str] = []

        self.characters : List[str] = list(set(list(''.join(self.name_reader.names))))
        self.start_character : str = '<S>'
        self.end_character : str = '<E>'

        if add_padding > 0 :
            self.characters += [self.start_character, self.end_character]
            for name in self.name_reader.names:
                padded_name = [self.start_character] * self.padding + list(name) + [self.end_character] * self.padding

                self.padded_names.append(padded_name)


        self.characters.sort()

        self.ctoi : Dict[str, int] = {c:index for index,c in enumerate(self.characters)}
        self.itoc : Dict[int, str]= {index:c for index,c in enumerate(self.characters)}

    def to_numpy(self) -> np.array:
        x = []

        for item in self.padded_names:
            for index in range(len(item) - self.padding):
                x.append(
                    [self.ctoi[item[index+p]] for p in range(self.padding + 1)]
                )

        
        return np.array(x).astype(np.int64)
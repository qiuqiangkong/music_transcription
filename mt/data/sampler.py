import random


class Sampler:
    def __init__(self, dataset_size: int):
        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)
        self.ptr = 0
        
    def __iter__(self) -> int:
        while True:
            if self.ptr == len(self.indexes):
                random.shuffle(self.indexes)
                self.ptr = 0
                
            index = self.indexes[self.ptr]
            self.ptr += 1

            yield index
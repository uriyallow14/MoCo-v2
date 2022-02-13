class DataLoaderCyclicIterator:
    def __init__(self, dataloader, load_labels=True):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.load_labels = load_labels

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.load_labels:
                return next(self.iterator)
            else:
                return next(self.iterator)[0]
        except StopIteration as e:
            self.iterator = iter(self.dataloader)
            raise e
            # return next(self)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random


class SampleDataset(Dataset):
    def __init__(self, r1, r2):
        randomlist = []
        for i in range(120):
            n = random.randint(r1, r2)
            randomlist.append(n)
        self.samples = randomlist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx])



if __name__ == '__main__':

    dataset = SampleDataset(1, 100)
    # print(dataset.__getitem__(5))


    loader = DataLoader(dataset,batch_size=1, shuffle=False)
    for i, batch in enumerate(loader):
        print(i, batch)
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_abs_file_paths(dir_name):
    for dir_path, _, filenames in os.walk(dir_name):
        for f in filenames:
            yield os.path.abspath(os.path.join(dir_path, f))


class NSTDataset(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform

        mode = 'train' if train else 'test'
        self.sampleA = [file for file in get_abs_file_paths(os.path.join(root_dir, f"{mode}A"))]
        self.sampleB = [file for file in get_abs_file_paths(os.path.join(root_dir, f"{mode}B"))]

    def __len__(self):
        return max(len(self.sampleA), len(self.sampleB))

    def __getitem__(self, idx):
        sampleA_len = len(self.sampleA)
        sampleB_len = len(self.sampleB)

        imageA = self.transform(Image.open(self.sampleA[idx % sampleA_len]))
        imageB = self.transform(Image.open(self.sampleB[idx % sampleB_len]))

        return {'imageA': imageA, 'imageB': imageB}

from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import os


class EmbeddingDataset(Dataset):
    def __init__(self, file_path="/home/work/YAI-Summer/YAICON/VGGnet/embedding/data"):
        self.ib_path = os.path.join(file_path, "ib_embeddings_each")
        self.t5_path = os.path.join(file_path, "t5_embeddings_each")
        self.text_len = 10000

    def __len__(self):
        return self.text_len

    def __getitem__(self, idx):

        ib = np.load(os.path.join(self.ib_path, "{0:07d}.npy".format(idx)))
        t5 = np.load(os.path.join(self.t5_path, "{0:07d}.npy".format(idx)))
        ib = np.expand_dims(ib, axis = 0)
        return ib,t5


if __name__ == "__main__":
    dataset = EmbeddingDataset()
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)

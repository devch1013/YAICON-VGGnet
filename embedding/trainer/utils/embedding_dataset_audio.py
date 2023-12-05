from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import os

class EmbeddingDataset(Dataset):
    def __init__(self, audio_file_path="/home/work/YAI-Summer/YAICON/VGGnet/data/audio"):
        self.audio_len = 2892
        self.ib_audio_path = os.path.join(audio_file_path, "audio_imagebind")
        self.t5_audio_path = os.path.join(audio_file_path, "caption_t5")

    def __len__(self):
        return self.audio_len

    def __getitem__(self, idx):
        ib = np.load(os.path.join(self.ib_audio_path, "audio_{}.npy".format(idx)))
        t5 = np.load(os.path.join(self.t5_audio_path, "caption_{}.npy".format(idx)))
        t5 = t5.squeeze(axis=0)
        return ib, t5

if __name__ == "__main__":
    dataset = EmbeddingDataset()
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)

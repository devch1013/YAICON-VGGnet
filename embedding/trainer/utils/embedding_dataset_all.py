from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import os


class EmbeddingDataset(Dataset):
    def __init__(self, file_path="/home/work/YAI-Summer/YAICON/VGGnet/embedding/data",
                 audio_file_path = "/home/work/YAI-Summer/YAICON/VGGnet/data/audio",
                 image_file_path = "/home/work/YAI-Summer/YAICON/VGGnet/data/image"):
        self.ib_path = os.path.join(file_path, "ib_embeddings_each")
        self.t5_path = os.path.join(file_path, "t5_embeddings_each")
        self.text_len = 10000
        self.audio_len = 2892
        self.image_len = 10000
        self.ib_audio_path = os.path.join(audio_file_path, "audio_imagebind")
        self.t5_audio_path = os.path.join(audio_file_path, "caption_t5")
        self.ib_image_path = os.path.join(image_file_path, "image")
        self.t5_image_path = os.path.join(image_file_path, "text")
        # self.ib_list = os.listdir(self.ib_path)
        # self.t5_list = os.listdir(self.t5_path)
        # self.data_len = min((len(self.ib_list), len(self.t5_list)))

    def __len__(self):
        # return 1034239
        return self.text_len + (self.audio_len * 4) + (self.image_len)
        # return self.audio_len

    def __getitem__(self, idx):

        if idx < self.text_len:
            ib = np.load(os.path.join(self.ib_path, "{0:07d}.npy".format(idx)))
            t5 = np.load(os.path.join(self.t5_path, "{0:07d}.npy".format(idx)))
            ib = np.expand_dims(ib, axis = 0)
        elif idx >= self.text_len and idx < self.text_len + (self.audio_len * 10):
            idx = (idx - self.text_len) % self.audio_len
            ib = np.load(os.path.join(self.ib_audio_path, "audio_{}.npy".format(idx)))
            t5 = np.load(os.path.join(self.t5_audio_path, "caption_{}.npy".format(idx)))
            t5 = t5.squeeze(axis = 0)
        elif idx >= self.text_len + (self.audio_len * 4):
            idx = (idx - (self.text_len + (self.audio_len * 10))) % self.image_len
            ib = np.load(os.path.join(self.ib_image_path, "image_{}.npy".format(idx)))
            t5 = np.load(os.path.join(self.t5_image_path, "text_{}.npy".format(idx)))
            t5 = t5.squeeze(axis = 0)
        # ib = np.load(os.path.join(self.ib_audio_path, "audio_{}.npy".format(idx)))
        # t5 = np.load(os.path.join(self.t5_audio_path, "caption_{}.npy".format(idx)))
        # t5 = t5.squeeze(axis = 0)
        return ib,t5



# # Example usage:
# file_path = './results_2M_val.csv'
# text_dataset = TextDataset(file_path)

# # Access text data line by line
# for i in range(len(text_dataset)):
#     text_line = text_dataset[i]
#     print(text_line)

if __name__ == "__main__":
    dataset = EmbeddingDataset()
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)

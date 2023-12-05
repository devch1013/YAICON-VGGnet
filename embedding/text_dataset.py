from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn



class TextDataset(Dataset):
    def __init__(self, file_path = "/home/work/YAI-Summer/YAICON/VGGnet/embedding/webvid/results_2M_train.csv"):
        self.data = []
        test = pd.read_csv(file_path)
        test = test['name']
        for i in test:
            self.data.append(i.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = TextDataset()
    for i in range(1000):
        print(dataset[i])   

    
    


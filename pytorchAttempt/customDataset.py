import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 
import numpy as np
from csv import reader


maximum = 140
num_input = 7
num_output = 4

def load_file(filename, max_rows):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        count = 0
        for row in datareader:
            if row is None:
                continue
            ret.append(row)
            count+=1
            if (max_rows > 0 and count >= max_rows):
                break
        return ret[1:]

class NutritionDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.data = load_file(csv_file, maximum)
        self.transform = transform 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_vals = self.data[idx][0:num_input]
        output_vals = self.data[idx][num_input:]
        
        sample = {"data": torch.from_numpy(np.float_(input_vals)).to(torch.float32), "label":torch.from_numpy(np.float_(output_vals).reshape(-1,1)).to(torch.float32)}

        #sample = torch.from_numpy(sample)
        #sample = {"data": torch.from_numpy(np.float_(input_vals).reshape(-1,1))).to(torch.float32), "label":torch.from_numpy(np.float_(output_vals).reshape(-1,1)).to(torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


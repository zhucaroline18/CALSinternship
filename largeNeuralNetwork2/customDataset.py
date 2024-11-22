import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 
import numpy as np
from csv import reader

#still not sure how to deal with null values 

maximum = 170
num_input = 87 #is this really the right number?
num_output = 36
labels = 5

def load_file(filename, max_rows):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        count = 0
        for row in datareader:
            if row is None:
                continue
            row2 = []
            for i, item in enumerate(row):
                if i >= labels:
                    if item != '':
                        row2.append(item)
                    else:
                        row2.append('0')
            ret.append(row2)
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
        
        sample = {"data": torch.from_numpy(np.float_(input_vals)).to(torch.float32), "label":torch.from_numpy(np.float_(output_vals)).to(torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample

class SingleLabelDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.data = load_file(csv_file, maximum)
        self.transform = transform 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_vals = self.data[idx][0:num_input]
        output_vals = self.data[idx][96:97]
        
        sample = {"data": torch.from_numpy(np.float_(input_vals)).to(torch.float32), "label":torch.from_numpy(np.float_(output_vals)).to(torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample

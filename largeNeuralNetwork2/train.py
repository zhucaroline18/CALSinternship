import torch
import torch.nn as nn 
import os.path 
import mymodel
import numpy as np
import customDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import pandas as pd
# from featurewiz import FeatureWiz


train_dataset = customDataset.SingleLabelDataset(csv_file = "largeNeuralNetwork2/OfficialTotalData.csv")
test_dataset = customDataset.SingleLabelDataset(csv_file = "largeNeuralNetwork2/OfficialTotalData.csv")


# i = 0
# data = []
# labels = []
# for d, l in train_dataset:
#     data.append(d)
#     labels.append(l)
#     i += 1
#     print(i)

# fwiz = FeatureWiz(feature_engg = '', nrows=None, transform_target=True, scalers="std",
#         		category_encoders="auto", add_missing=False, verbose=0, imbalanced=False, 
#                 ae_options={})
# X_train_selected, y_train = fwiz.fit_transform(data, labels)
# X_test_selected = fwiz.transform(data)
### get list of selected features ###
# print(fwiz.features)

global_loss = nn.MSELoss() #next try cross entropy
testing_loss = nn.L1Loss()
largeLearningRate = 0.01
fineTuneLearningRate = 0.0001
globalBatchSize = 1 #8 or 16 or 32
wholeTrain = 100

global_path = "journal_club.pth"
checkpoint_path = "journal_club_checkpoint.pth"

def train_model(path):
    global_path = path + ".pth"
    checkpoint_path = path + "_checkpoint.pth"
    model = mymodel.MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = largeLearningRate)
    loss_fn = global_loss
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = globalBatchSize, shuffle = True)

    for epoch in range(wholeTrain):
        for i, (obj) in enumerate(train_loader):
            data = (obj['data'][0])
            label = (obj['label'])
            if label == 0:
                continue
            #print(label)
    

            data.to(torch.float32)
            label.to(torch.float32)
            #print(data)
            
            y = model(data)
            loss = loss_fn(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i%10 == 0):
                print(f"epoch {epoch}: loss = {loss}")

    torch.save(model, global_path)

    state = {
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict()
    }
    torch.save(state, checkpoint_path)

def load_checkpoint_train_model(path):
    global_path = path + ".pth"
    checkpoint_path = path + "_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    model = mymodel.MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = fineTuneLearningRate)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_fn = global_loss  #maybe we can do cross entropy to get rid of nois?

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = globalBatchSize, shuffle = True)
    for epoch in range(wholeTrain):
        for i, (obj) in enumerate(train_loader):
            data = (obj['data'][0])
            label = (obj['label'])

            data.to(torch.float32)
            label.to(torch.float32)

            y = model(data)
            loss = loss_fn(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"epoch {epoch}: loss = {loss}")

    torch.save(model, global_path)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, checkpoint_path)

def test_model_all(path):
    global_path = path + ".pth"
    model = torch.load(global_path)

    total = 0
    totalLoss = 0
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = globalBatchSize, shuffle = False)

    for i, (obj) in enumerate(test_loader):
        data = obj['data'][0]
        label = obj['label'][0]

        if label == 0:
            continue

        data.to(torch.float32)
        label.to(torch.float32)

        y = model(data)
        #loss = global_loss(y, label)
        loss2 = testing_loss(y, label)
        totalLoss += loss2
        
        print(f"loss: {loss2}")
        #print(f"input: {data.detach().cpu().numpy()}")
        print(f"result: {y.detach().cpu().numpy()}")
        print(f"label: {label.detach().cpu().numpy().reshape(-1)}")
        print()
        total += 1
    return totalLoss/total

    '''for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        output = model(data)
        total += (np.sum(label-total)*np.sum(label-total))
        print(f"label: {label}")
        print(f"output: {output}")

        print(f'RMSE is {total/len(test_dataset)}')'''

if __name__ == "__main__":
    train_model()
    load_checkpoint_train_model()
    loss = 0
    for x in range(100):
        loss += test_model_all()
    print(f"average loss: {loss/100}")
    

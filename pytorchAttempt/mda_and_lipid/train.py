import torch
import torch.nn as nn 
import mymodel
import numpy as np
import customDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

train_dataset = customDataset.NutritionDataset(csv_file = "trainingdata.csv")
test_dataset = customDataset.NutritionDataset(csv_file = "trainingdata.csv")

global_loss = nn.MSELoss() #next try cross entropy
testing_loss = nn.L1Loss()
#global_loss = nn.CrossEntropyLoss()
largeLearningRate = 0.0001
fineTuneLearningRate = 0.0001
globalBatchSize = 1 #8 or 16 or 32
wholeTrain = 20

#test the model effectiveness
#databases 

global_path = "pytorch_attempt.pth"
checkpoint_path = "pytorch_attempt_checkpoint.pth"

def train_model():
    model = mymodel.MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = largeLearningRate)
    loss_fn = global_loss
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

            if (i%10 == 0):
                print(f"epoch {epoch}: loss = {loss}")

    torch.save(model, global_path)

    state = {
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict()
    }
    torch.save(state, checkpoint_path)

def load_checkpoint_train_model():
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

def test_model_all():
    model = torch.load(global_path)

    total = 0
    totalLoss = 0
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = globalBatchSize, shuffle = False)

    for i, (obj) in enumerate(test_loader):
        data = obj['data'][0]
        label = obj['label'][0]

        data.to(torch.float32)
        label.to(torch.float32)

        y = model(data)
        #loss = global_loss(y, label)
        loss2 = testing_loss(y, label)
        totalLoss += loss2
        
        print(f"loss: {loss2}")
        print(f"input: {data.detach().cpu().numpy()}")
        print(f"result: {y.detach().cpu().numpy()}")
        print(f"label: {label.detach().cpu().numpy().reshape(-1)}")
        print()
        total += 1
    print(f"average loss: {totalLoss/total}")

    '''for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        output = model(data)
        total += (np.sum(label-total)*np.sum(label-total))
        print(f"label: {label}")
        print(f"output: {output}")

        print(f'RMSE is {total/len(test_dataset)}')'''

if __name__ == "__main__":
    train_model()
    #load_checkpoint_train_model()
    #load_checkpoint_train_model()
    #print(train_dataset)
    test_model_all()
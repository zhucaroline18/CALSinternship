import torch
import torch.nn as nn 
import mymodel
import numpy as np
import customDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import math
import time
from datetime import timedelta
from pathlib import Path
# from featurewiz import FeatureWiz


train_dataset = customDataset.SingleLabelDataset(csv_file = "smallModel/Calculated_Value_Dataset.csv")
test_dataset = customDataset.SingleLabelDataset(csv_file = "smallModel/Calculated_Value_Dataset.csv")


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

mse_loss = nn.MSELoss() #next try cross entropy
abs_loss = nn.L1Loss()
cross_entropy_loss = nn.CrossEntropyLoss()
largeLearningRate = 0.01
fineTuneLearningRate = 0.0001
globalBatchSize = 1 #8 or 16 or 32
wholeTrain = 100


label_map = ["average feed intake","bodyweightgain","akp","alt",
            "gluclose","nefa","pip","tc","tg","trap","uric acid","BCA","breast mTOR",
            "breast S6K1","breast 4E-BP1","breast MURF1","breast MAFbx","breast AMPK",
            "breast LAT1","breast CAT1","breast SNAT2","breast VDAC1","breast ANTKMT",
            "breast AKT1","IGF1","IGFR","IRS1","FOXO1","LC3-1","MyoD","MyoG","Pax3","Pax7","Mrf4",
            "Mrf5","liver mTOR","liver S6K1","liver 4E-BP1","liver MURF1","liver MAFbx",
            "liver AMPK","breast weight (g)","breast PH","breast WHC",
            "breast HARDNESS","breast SPRINGINESS","breast CHEWINESS",
            "breast COHESIVENESS","breast GUMMINESS","breast RESILIENCE",
            "Thigh Weight (g)","thigh PH","thigh WHC","thigh HARDNESS",
            "thigh SPRINGINESS","thigh CHEWINESS","liver weight(g)","Plasma SFA",
            "Plasma MUFA","Plasma PUFA","Plasma n-3","Plasma n-6","Plasma C18:3 ",
            "Plasma C20:5","Plasma C22:6","Liver SFA","Liver MUFA","Liver PUFA",
            "Liver n-3","Liver n-6","Liver C18:3 ","Liver C20:5","Liver C22:6",
            "Breast SFA","Breast MUFA","Breast PUFA","Breast n-3","Breast n-6",
            "Breast C18:3","Breast C20:5","Breast C22:6","Thigh SFA","Thigh MUFA",
            "Thigh PUFA","Thigh n-3","Thigh n-6","Thigh C18:3 ","Thigh C20:5","Thigh C22:6"]

def train_model(dataset, path):
    model = mymodel.MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = largeLearningRate)
    loss_fn = mse_loss
    train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = globalBatchSize, shuffle = True)

    for epoch in range(wholeTrain):
        for i, (obj) in enumerate(train_loader):
            data = (obj['data'][0])
            label = (obj['label'][0])
            if label.numel() == 0 or label == 0:
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

            #if (i%10 == 0):
            #    print(f"epoch {epoch}: loss = {loss}")

    torch.save(model, path + ".pth")

    state = {
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict()
    }
    torch.save(state, path + "_checkpoint.pth")
    load_checkpoint_train_model(dataset, path)

def load_checkpoint_train_model(dataset, path):
    checkpoint = torch.load(path + "_checkpoint.pth")
    model = mymodel.MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = fineTuneLearningRate)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_fn = mse_loss  #maybe we can do cross entropy to get rid of nois?

    train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = globalBatchSize, shuffle = True)
    for epoch in range(wholeTrain):
        for i, (obj) in enumerate(train_loader):
            data = (obj['data'][0])
            label = (obj['label'][0])

            data.to(torch.float32)
            label.to(torch.float32)

            y = model(data)
            loss = loss_fn(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if i % 10 == 0:
            #    print(f"epoch {epoch}: loss = {loss}")

    torch.save(model, path + ".pth")

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path + "_checkpoint.pth")

def test_model_all(dataset:customDataset, path):
    model = torch.load(path + ".pth")

    total = 0
    APE_agr = 0
    SE_agr = 0
    CEL_agr = 0
    res2_agr = 0
    sqr_agr = 0

    test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = globalBatchSize, shuffle = False)

    for i, (obj) in enumerate(test_loader):
        data = obj['data'][0]
        label = obj['label'][0]

        if label.numel() == 0 or label == 0:
            continue

        data.to(torch.float32)
        label.to(torch.float32)

        y = model(data)
        #loss = global_loss(y, label)
        diff = abs_loss(y, label)
        APE_agr += (diff/label).item()
        SE_agr += mse_loss(y, label).item()
        CEL_agr += torch.exp(-1 * cross_entropy_loss(y, label)).item()
        res2_agr += (diff**2).item()
        sqr_agr += (label**2).item()
        total += 1

        
        #print(f"loss: {loss2}")
        #print(f"input: {data.detach().cpu().numpy()}")
        #print(f"result: {y.detach().cpu().numpy()}")
        #print(f"label: {label.detach().cpu().numpy().reshape(-1)}")
        #print()
        
    if total == 0:
        total = 1
    if CEL_agr == 0:
        CEL_agr = 1

    if sqr_agr == 0:
        sqr_agr = 1
    
    k = customDataset.num_input
    likelihood = CEL_agr/total
    n = total



    MAPE = APE_agr/total
    AIC = 2 * k - 2 * math.log(likelihood) 
    BIC = k * math.log (n) - 2 * math.log(likelihood)
    R2 = 1 - res2_agr/sqr_agr
    RMSE = math.sqrt(SE_agr)/total

    return MAPE, AIC, BIC, R2, RMSE

    '''for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        output = model(data)
        total += (np.sum(label-total)*np.sum(label-total))
        print(f"label: {label}")
        print(f"output: {output}")

        print(f'RMSE is {total/len(test_dataset)}')'''

if __name__ == "__main__":

    output = ""
    output_file = f"outputs/training_run_{time.time()}"
    Path("smallModel/" + output_file).mkdir()
    start = time.time()
    last = start

    for i in range(88):

        dataset = customDataset.NutritionDataset("smallModel/Calculated_Value_Dataset.csv", i + 93)
        train_model(dataset, f"smallModel/{output_file}/{label_map[i]}")
        loss = np.zeros(5)

        for x in range(100):
            metrics = np.array(test_model_all(dataset, f"smallModel/{output_file}/{label_map[i]}"))
            loss += metrics

        loss /= 100

        output += f"""{label_map[i]}
              MAPE: {loss[0]}]
              AIC: {loss[1]}]
              BIC: {loss[2]}]
              R^2: {loss[3]}]
              RMSE: {loss[4]}]\n
              """
        
        print(f"""{timedelta(seconds=time.time()-start)}: model for {label_map[i]} completed in {timedelta(seconds=time.time()-last)}""")
        last = time.time()

        
    with open("results.txt", "w") as file:
        file.write(output)

    print(f"total time elapsed: {timedelta(seconds=time.time()-start)}")
    
    

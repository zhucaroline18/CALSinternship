import torch 
import torch.nn as nn

# not doing any flask things right now- that's for deployment later

def test_with_user_input():
    # TODO: train the model to this path 
    model = torch.load("my_model.pth")
    input = 8
    #TODO: figure out how to format input etc
    output = model(input)
    predict = #format the output
    print(f'predict:{predict}')

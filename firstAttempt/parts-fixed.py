# divide result values by 162.42, 3.61, 8.54, 2.824388, 42.65
import numpy as np
from csv import reader

num_input = 7
num_output = 1
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
        return ret
    
def load_dataset(filename, max_rows):
    if (max_rows > 0):
        max_rows += 1
    data = load_file(filename, max_rows)
    data = data[1:]
    train_dataset = list()
    test_dataset = list()
    train_labels = list()
    test_labels = list()
    for raw_row in data:
        var = [raw_row[0:num_input]]
        row = [raw_row[num_input:]]

        inTraining=True
        for item in row[0]:
            if item == '':
                test_dataset.append(var)
                test_labels.append(row)
                inTraining = False
        if inTraining:
            for item in var[0]:
                if item == '':
                    test_dataset.append(var)
                    test_labels.append(row)
                    inTraining = False
        if inTraining:
            train_dataset.append(var)
            train_labels.append(row)
    return test_dataset, test_labels, train_dataset, train_labels

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    toReturn = list()
    for row in Z:
        new_row = list()
        for col in row:
            new_col = 1 if col > 0 else 0
            new_row.append(new_col)
        toReturn.append(new_row)
    return toReturn

def leakyReLU(Z):
    return np.maximum (Z, 0.01*Z)

def leakyReLU_deriv(Z):
    toReturn = list()
    for row in Z:
        new_row = list()
        for col in row:
            new_col = 1 if col > 0 else 0.01
            new_row.append(new_col)
        toReturn.append(new_row)
    return toReturn

def softmax(Z):
    exp = np.exp(Z)
    sum = np.sum(exp)
    return exp/sum

class NeuralNetwork:
    def __init__(self, inputCount, hiddenCount, hidden2Count, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.hiddenCount = hiddenCount
        self.hidden2Count = hidden2Count

        self.W1 = np.random.rand(hiddenCount, inputCount) - 0.5
        self.b1 = np.random.rand(hiddenCount, 1) - 0.5
        self.W2 = np.random.rand(hidden2Count, hiddenCount)-0.5
        self.b2 = np.random.rand(hidden2Count, 1) - 0.5
        self.W3 = np.random.rand(outputCount, hidden2Count)-0.5
        self.b3 = np.random.rand(outputCount, 1) - 0.5
        self.learning_rate = 0.005

    def feed_forward(self, X):
        #X is the input 
        Z1 = self.W1.dot(X) + self.b1
        A1 = leakyReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = leakyReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = Z3
        return Z1, A1, Z2, A2, Z3, A3
    
    def backward_prop(self, X, Z1, A1, Z2, A2, Z3, A3, Y):
        dZ3 = A3-Y # output the loss to see if it decreases
        dW3 = dZ3.dot(A2.T)
        db3 = dZ3
        dZ2 = self.W3.T.dot(dZ3)*leakyReLU_deriv(Z2)
        dW2 = dZ2.dot(A1.T)
        db2 = dZ2
        dZ1 = self.W2.T.dot(dZ2) * leakyReLU_deriv(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = dZ1

        self.W1 = self.W1 - self.learning_rate * dW1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.W3 = self.W3 - self.learning_rate * dW3
        self.b1 = self.b1 - self.learning_rate * db1
        self.b2 = self.b2 - self.learning_rate * db2
        self.b3 = self.b3 - self.learning_rate * db3

    def train (self, X, Y):
        Z1, A1, Z2, A2, Z3, A3 = self.feed_forward(X)
        self.backward_prop(X, Z1, A1, Z2, A2, Z3, A3, Y)

    def predict(self, X):
        Z1, A1, Z2, A2, Z3, A3 = self.feed_forward(X)
        self.formatOutput(X, A3)
        return A3

    def formatOutput(self, input, A2):
        print('input:')
        print(f'chicken part: {"breast" if input[0] == 2.5 else ("thigh" if input[0] == 5 else ("liver" if input[0]==7.5 else "adipose"))}')
        print(f'concentration: {"0%" if input[1] == 0 else ("1%" if input[1] == 0.25 else ("2%" if input[1]==0.5 else "4%"))}')
        print(f'week: {"3" if input[2]==1 else "6"}')
        print(f'bca/weight: {input[3]}')
        print(f'MDA/WEIGHT: {input[4] }')
        print(f'NEFA/WEIGHT: {input[5] }')
        print(f'TC/WEIGHT: {input[6]}')

        print ('ouput:')
        #print(f'BCA/WEIGHT: {A2[0] * 162.42}')
        
        print(f'TG/WEIGHT: {A2[0] }')
        print()

def train_and_predict():
    #TODO: UPDATE VALUES 
    brain = NeuralNetwork(num_input, 4, 4, num_output)
    test_dataset, test_labels, train_dataset, train_labels = load_dataset('chickenparts.csv', 200)
    for i in range(len(train_dataset)):
        X = list(np.float_(train_dataset[i][0]))
        X = np.array(X).reshape(-1,1)
        Y = list(np.float_(train_labels[i][0]))
        Y = np.array(Y).reshape(-1,1)

        toDivideBy = np.array([3.61, 8.54, 2.824388, 42.65]).reshape(-1, 1)
        Y2 = Y
        brain.train(X, Y2)

    differenceSum = 0
    for i in range(len(test_dataset)) :
        X = list(np.float_(train_dataset[i][0]))
        X = np.array(X).reshape(-1,1)
        prediction = brain.predict(X)
        Y = list(np.float_(train_labels[i][0]))
        Y = np.array(Y).reshape(-1,1)
        toDivideBy = np.array([3.61, 8.54, 2.824388, 42.65]).reshape(-1, 1)
        Y2 = Y
        sumArray = Y2-prediction
        result = np.sum(sumArray)
        exp = result*result
        differenceSum += exp

    print(f'RMSE is {differenceSum/len(test_dataset)}')
    return brain
        
def train_again(brain):
    test_dataset, test_labels, train_dataset, train_labels = load_dataset('chickenparts.csv', 200)
    for i in range(len(train_dataset)):

        X = list(np.float_(train_dataset[i][0]))
        X = np.array(X).reshape(-1,1)
        Y = list(np.float_(train_labels[i][0]))
        Y = np.array(Y).reshape(-1,1)

        toDivideBy = np.array([3.61, 8.54, 2.824388, 42.65]).reshape(-1, 1)
        Y2 = Y
        brain.train(X, Y2)

if __name__ == "__main__":
    #brain, w1, w2, b1, b2 = train_and_predict()
    #train_again(brain)
    #print(train_labels)
    #show_data_at(dataset, labels, 77)
    brain = train_and_predict()
    train_again(brain)
    train_again(brain)
    train_again(brain)
    train_again(brain)
    train_and_predict()


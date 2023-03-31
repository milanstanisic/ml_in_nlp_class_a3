import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch import optim
from sklearn.metrics import precision_score, recall_score
# Whatever other imports you need

# You can implement classes and helper functions here too.
class Dataset():
    def __init__(self, data):
        vectors = data['vector']
        labels = data['label']
        self.X = torch.tensor(vectors, dtype = torch.float32)
        self.y = torch.tensor(labels, dtype =  torch.float32)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument("hiddensize", type=int, help="Size of the hidden layer", default = 0)
    parser.add_argument("non_linearity", type=str, help="The activation function for the hidden layer. Possible values: sigmoid, tanh, relu")
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    # implement everything you need here
    
    #Label encoding 
    print("Encoding labels")
    data = pd.read_csv(args.featurefile)
    labels = pd.DataFrame(data['label'].unique(), columns = ['label'])
    labels['id'] = pd.Series(range(len(labels)))
    labels.set_index('label', inplace = True, drop = False)
    data['label'] = data['label'].apply(lambda x: labels.at[x, 'id'])
    binarizer = LabelBinarizer()
    binarizer.fit(range(len(labels)))
    data['label'] = data['label'].apply(lambda x: binarizer.transform([x])[0])
    
    
    #Preparing the tensors
    data['vector'] = data['vector'].apply(lambda x: [float(y) for y in x[1:-1].split()])
    training = Dataset(data[data['subset'].isin(['train'])].reset_index(drop = True)[['label', 'vector']])
    train_loader = torch.utils.data.DataLoader(training, batch_size = 10, shuffle = True)
    test = Dataset(data[data['subset'].isin(['test'])].reset_index(drop = True)[['label', 'vector']])
    test_loader = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

    
    #Defining the model
    input_size = len(data.at[0, 'vector'])
    non_linearity_dict = {"sigmoid":nn.Sigmoid(), "tanh":nn.Tanh(), "relu":nn.ReLU()}
    chosen_act = non_linearity_dict[args.non_linearity]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential()
    if args.hiddensize != 0:
        model.add_module('Hidden', nn.Linear(in_features = input_size, out_features = args.hiddensize, bias = True))
        model.add_module('Activation_hidden', chosen_act)
    model.add_module('Layer_basic', nn.Linear(in_features = input_size if args.hiddensize == 0 else args.hiddensize, out_features = len(labels), bias = True))
    model.add_module('Activation_basic', nn.Softmax())
    model.to(device)
    
    optimizer = optim.RMSprop(model.parameters(), lr=0.001) 
    loss_function = nn.MSELoss()

    print("Training the model")
    assigned_train = []
    for epoch in range(20):  
        running_loss = 0.0
        assigned_train = []
        for X, y in train_loader:
            optimizer.zero_grad()
            model.train()
            outputs = model.forward(X)
            assigned_train.extend(outputs.detach().numpy().tolist())
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {}, loss: {}'.format(epoch+1, running_loss))
    print('Finished Training')
    
    print("Evaluating the model")
    model.eval()
    model.train(False)
    assigned_test = []
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model.forward(X)
            assigned_test.extend(outputs.detach().numpy().tolist())
            loss = loss_function(outputs, y)
            test_loss += loss.item()
    print("Running loss: {}".format(test_loss))
    
    assigned_train = pd.DataFrame({"assigned":assigned_train})
    assigned_train['assigned'] = assigned_train['assigned'].apply(lambda x: x.index(max(x)))
    assigned_train['actual'] = training.y.tolist()
    assigned_train['actual'] = assigned_train['actual'].apply(lambda x: x.index(max(x)))
    
    assigned_test = pd.DataFrame({"assigned":assigned_test})
    assigned_test['assigned'] = assigned_test['assigned'].apply(lambda x: x.index(max(x)))
    assigned_test['actual'] = test.y.tolist()
    assigned_test['actual'] = assigned_test['actual'].apply(lambda x: x.index(max(x)))

    conf_matrix = np.zeros((len(labels), len(labels)))
    for row in assigned_test.itertuples():
        conf_matrix[row.assigned, row.actual] += 1
    conf_matrix = pd.DataFrame({labels.index[x]:conf_matrix[x] for x in range(conf_matrix.shape[0])})
    conf_matrix.set_index(labels.index, inplace = True)
    print(conf_matrix.to_markdown())
    print("Micro precision (test set):", precision_score(assigned_test['actual'], assigned_test['assigned'], average = 'micro'))
    print("Micro recall (test set):", recall_score(assigned_test['actual'], assigned_test['assigned'], average = 'micro'))
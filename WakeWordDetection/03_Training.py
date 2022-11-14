####### IMPORTS #############
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
import itertools
import seaborn as sns
warnings.filterwarnings("ignore")
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from plot_cm import plot_confusion_matrix

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")

####### Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)

y = np.array(df["class_label"].tolist())

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##### Prepare data : Pytorch ############
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 64

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


n_input_dim = X_train.shape[1]

# Layer size
n_hidden1 = 30  # Number of hidden nodes
n_hidden2 = 10
n_output =  1   # Number of output nodes = for binary classifier

##### Simple ANN ############

class ClfModel(nn.Module):
    def __init__(self):
        super(ClfModel, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output) 
        
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        # self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        # self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        # x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
model = ClfModel()
print(model)

##### Training ############

learning_rate = 0.01
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 500
loss_values = []

model.train()
for epoch in range(num_epochs):
    for X, y in train_dataloader:
        y_pred = model(X)            # Forward Propagation
        loss = loss_fn(y_pred, y.unsqueeze(-1))  # Loss Computation
        optimizer.zero_grad()         # Clearing all previous gradients, setting to zero 
        loss.backward()               # Back Propagation
        optimizer.step()              # Updating the parameters 
    loss_values.append(loss.item())

#### Evaluating our model ###########
y_pred = []
y_test = []
total, correct = 0, 0
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()
print(f'Accuracy test instances: {100 * correct // total}%')

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = list(itertools.chain(*y_pred))
y_test = list(itertools.chain(*y_test))
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])


#### Saving our model ###########
torch.save(model.state_dict(), 'saved_model/model.pt')
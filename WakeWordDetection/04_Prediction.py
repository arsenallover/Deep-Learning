######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import torch
from torch import nn

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

####### ALL CONSTANTS #####
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

n_hidden1 = 30  # Number of hidden nodes
n_hidden2 = 10
n_output =  1
n_input_dim = 40

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
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
model = ClfModel()

##### LOADING OUR SAVED MODEL and PREDICTING ###
model.load_state_dict(torch.load("saved_model/model.pt"))

print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    mfcc_processed = torch.from_numpy(mfcc_processed).reshape(1, -1)
    prediction = model(mfcc_processed)
    if prediction.item() > 0.9:
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction.item())
        i += 1
    
    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction.item())

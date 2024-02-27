import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

##### Utils functions 
def sigmoid(s):
    return 1 / (1 + torch.exp(-s))

def sigmoid_derivative(s):
    # derivative of sigmoid
    # s: sigmoid output
    return s * (1 - s)

def tanh(t):
    return torch.div(torch.exp(t) - torch.exp(-t), torch.exp(t) + torch.exp(-t))

def tanh_derivative(t):
    # derivative of tanh
    # t: tanh output
    return 1 - t * t

class Neural_Network:
    def __init__(self, input_size=200, output_size=1, hidden_size=6):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.b1 = torch.zeros(self.hiddenSize)
        
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)
        
    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = tanh(self.z1) 
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return sigmoid(self.z2)

    
    def backward(self, X, y, y_hat, lr=.1):
        
        batch_size = y.size(0)
        dl_dy_hat = (1/batch_size)*((y_hat - y)/ (y_hat * (torch.ones(y_hat.shape[-1]) - y_hat))) 
        dl_dz2 =  dl_dy_hat * sigmoid_derivative(sigmoid(self.z2))

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * tanh_derivative(self.h)
        
        self.W1 -= lr*torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr*torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr*torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr*torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))
    
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)


class FeedForwardNN(nn.Module):

    def __init__(self, vocab_size, num_classes, hidden_dim=100):
        super(FeedForwardNN, self).__init__()
        self.first_layer = nn.Linear(vocab_size, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss
    

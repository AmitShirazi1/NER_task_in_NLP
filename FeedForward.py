import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    return torch.div(torch.exp(x) - torch.exp(-x), torch.exp(x) + torch.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)


def d_tanh(x):
    return 1 - x * x

class FeedForward:
    def __init__(self, input_size, output_size, hidden_size1, lr=0.01):
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size1 = hidden_size1

        # weights
        self.W1 = torch.randn(self.input_size, self.hidden_size1)
        self.b1 = torch.zeros(self.hidden_size1)

        self.W2 = torch.randn(self.hidden_size1, self.output_size)
        self.b2 = torch.zeros(self.output_size)

        self.lr = lr

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = tanh(self.z1) 
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return sigmoid(self.z2)

        
    def backward(self, X, y, y_hat, lr=.1):
        batch_size = y.size(0)
        dl_dy_hat = (1/batch_size)*((y_hat - y)/ (y_hat * (torch.ones(y_hat.shape[-1]) - y_hat))) 
        dl_dz2 =  dl_dy_hat * d_sigmoid(sigmoid(self.z2))

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * d_tanh(self.h)
        
        self.W1 -= lr*torch.matmul(torch.unsqueeze(X,1), dl_dz1)
        self.b1 -= lr*torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr*torch.matmul(torch.unsqueeze(self.h,1), dl_dz2)
        self.b2 -= lr*torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

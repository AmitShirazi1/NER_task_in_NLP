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
    
    def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                        "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
        model.to(device)

        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = 0.0

                for batch in data_loaders[phase]:
                    batch_size = 0
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                        batch_size = v.shape[0]

                    optimizer.zero_grad()
                    if phase == 'train':
                        outputs, loss = model(**batch)
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            outputs, loss = model(**batch)
                    pred = outputs.argmax(dim=-1).clone().detach().cpu()

                    cur_num_correct = accuracy_score(batch['labels'].cpu().view(-1), pred.view(-1), normalize=False)

                    running_loss += loss.item() * batch_size
                    running_acc += cur_num_correct

                epoch_loss = running_loss / len(data_sets[phase])
                epoch_acc = running_acc / len(data_sets[phase])

                epoch_acc = round(epoch_acc, 5)
                if phase.title() == "test":
                    print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
                else:
                    print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    with open('model.pkl', 'wb') as f:
                        torch.save(model, f)
            print()

        print(f'Best Validation Accuracy: {best_acc:4f}')
        with open('model.pkl', 'rb') as f:
            model = torch.load(f)
        return model
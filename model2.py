import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np

from datasets_classes import CustomDataset, open_and_split_file

from gensim import downloader

from lstm_classes import train

class FeedForwardNN(nn.Module):
    """
    FeedForward nural network
    """
    def __init__(self, vec_dim, num_classes, hidden_dim=100):
        super(FeedForwardNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim) # linear layer 1
        self.second_layer = nn.Linear(hidden_dim, num_classes) # linear layer 2
        self.activation = nn.ReLU() # activation function
        self.sigmoid = nn.Sigmoid() # sigmoid
        self.loss = nn.CrossEntropyLoss() # loss function

    def forward(self, word, labels=None):
        """
        @param word -> vector representation of word
        @parma labels -> the label of a word if forward is used on train else None
        """
        x = self.first_layer(word)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.sigmoid(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss



def model2(train_path, dev_path):
    """
    Prepares the data, and runs the train functions
    """
    train_dataset = open_and_split_file(train_path, glove_twitter, vec_num)
    train_dataset = CustomDataset(train_dataset[0], train_dataset[1])
    dev_dataset = open_and_split_file(dev_path, glove_twitter, vec_num)
    dev_dataset = CustomDataset(dev_dataset[0], dev_dataset[1])
    datasets = {"train": train_dataset, "dev": dev_dataset}

    model = FeedForwardNN(vec_num, 2, hidden_dim=int(vec_num*2))
    optimizer = Adam(params=model.parameters())
    print("-----------------Model 2-----------------")
    predictions = train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=15)
    print("----------------------------------------")


if __name__ == "__main__":
    vec_num = 100
    GLOVE_PATH = f'glove-twitter-{vec_num}'
    glove_twitter = downloader.load(GLOVE_PATH)

    model2("/home/student/hw2/NER_task_in_NLP/data/train.tagged", "/home/student/hw2/NER_task_in_NLP/data/dev.tagged")

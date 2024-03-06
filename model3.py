import torch.nn as nn
from torch.optim import Adam

from gensim import downloader

from lstm_classes import model3_LSTM, train
from datasets_classes import CustomDataset, open_and_split_file


def model3(train_path, dev_path):
    """
    Prepares the data, and runs the train functions
    """
    num_epochs = 15
    learning_rate = 0.001

    num_stacked_layers = 1  # Number of stacked lstm layers, in this model, we do not stack layers.
    num_classes = 2  # Number of output classes

    lstm = model3_LSTM(num_classes, vec_num, int(vec_num/2), vec_num*2, num_stacked_layers)  # Initiate the model

    optimizer = Adam(lstm.parameters(), lr=learning_rate)  # Adam optimizer\

    train_dataset = open_and_split_file(train_path, glove_twitter, vec_num)
    train_dataset = CustomDataset(train_dataset[0], train_dataset[1])
    dev_dataset = open_and_split_file(dev_path, glove_twitter, vec_num)
    dev_dataset = CustomDataset(dev_dataset[0], dev_dataset[1])
    datasets = {"train": train_dataset, "dev": dev_dataset}

    print("-----------------Model 3-----------------")
    predictions = train(model=lstm, data_sets=datasets, optimizer=optimizer, num_epochs=num_epochs)
    print("----------------------------------------")


if __name__ == "__main__":
    vec_num = 100
    GLOVE_PATH = f'glove-twitter-{vec_num}'
    glove_twitter = downloader.load(GLOVE_PATH)
    model3("/home/student/hw2/NER_task_in_NLP/data/train.tagged", "/home/student/hw2/NER_task_in_NLP/data/dev.tagged")

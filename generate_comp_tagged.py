import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim import Adam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np

from model4 import model4

TRAIN_PATH = "/home/student/hw2/NER_task_in_NLP/data/train.tagged"
DEV_PATH = "/home/student/hw2/NER_task_in_NLP/data/dev.tagged"
TEST_PATH = "/home/student/hw2/NER_task_in_NLP/data/test.untagged"


if __name__ == "__main__":
    sentences, predictions, lstm_model = model4(TRAIN_PATH, DEV_PATH, TEST_PATH)
    with open("comp_314779166_325549681.tagged", "w") as f:
        for sentence in sentences:
            for word in sentence:
                f.write(word + "\t" + ("O" if predictions[0].item() == 0 else "1") + "\n")
                predictions = predictions[1:]
            f.write("\n")

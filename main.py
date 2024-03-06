from gensim import downloader
import numpy as np

from model1 import model1
from model2 import model2
from model3 import model3


vec_num = 100
GLOVE_PATH = f'glove-twitter-{vec_num}'
glove_twitter = downloader.load(GLOVE_PATH)


def open_and_split_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        words = []
        tags = []
        for line in lines:
            try:
                word, tag = line.rstrip().split("\t")
                word = word.lower()
                if (word not in glove_twitter):
                    words.append(np.zeros(vec_num))
                else:
                    words.append(glove_twitter[word])
                tags.append(0 if tag == "O" else 1)

            except:
                continue
    return words, tags

train_words, train_labels = open_and_split_file("/home/student/hw2/NER_task_in_NLP/data/train.tagged")
dev_words, dev_labels = open_and_split_file("/home/student/hw2/NER_task_in_NLP/data/dev.tagged")

model1(train_words, train_labels, dev_words, dev_labels)

model2('/home/student/hw2/NER_task_in_NLP/data/train.tagged', "/home/student/hw2/NER_task_in_NLP/data/dev.tagged")

model3("/home/student/hw2/NER_task_in_NLP/data/train.tagged", "/home/student/hw2/NER_task_in_NLP/data/dev.tagged")

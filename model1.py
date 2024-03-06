from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from gensim import downloader
import numpy as np

def open_and_split_file(file_path):
    """
    splits the file and creates retuns two lists
    @param words: a list of GLoVe vectors of words in the file (if the word doesn't exists in GLoVe vocab return np.zeros(vec_num))
    @param tags: a list of tags in the file, such that tags[i] is tag of words[i]
    """
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

def model1(train_words, train_labels, dev_words, dev_labels):
    """
    train a knn model on train_words and train_labels,
    predicts on dev_words,
    calculates the f1 score on dev
    """
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(train_words, train_labels)
    y_pred = knn.predict(dev_words)
    print("-----------------Model 1-----------------")    
    print("F1 score on train: ", f1_score(train_labels, knn.predict(train_words)))
    print("F1 score on dev: ", f1_score(dev_labels, y_pred))
    print("----------------------------------------")


if __name__ == "__main__":
    vec_num = 100 # number of features in word vector
    GLOVE_PATH = f'glove-twitter-{vec_num}'
    glove_twitter = downloader.load(GLOVE_PATH)

    train_words, train_labels = open_and_split_file("/home/student/hw2/NER_task_in_NLP/data/train.tagged")
    dev_words, dev_labels = open_and_split_file("/home/student/hw2/NER_task_in_NLP/data/dev.tagged")

    model1(train_words, train_labels, dev_words, dev_labels)
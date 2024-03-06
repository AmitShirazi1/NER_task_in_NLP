from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np

class CompetativeDataset(Dataset):
    """
    A class represnting a dataset.
    On getitem returns a dictionary = {"word": words[index], "labels": tags[index]}
    """
    def __init__(self, sentences, labels):
        # Create a path-to-label dictionary
        self.sentences = sentences
        self.tags = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = torch.from_numpy(self.sentences[index]).squeeze()
        tag = self.tags[index]
        data = {"sentence": sentence, "labels": tag}
        return data


class CustomDataset(Dataset):
    """
    A class represnting a dataset.
    On getitem returns a dictionary = {"word": words[index], "labels": tags[index]}
    """
    def __init__(self, words, tags):
        # Create a path-to-label dictionary
        self.words, self.tags = words, tags

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word = np.array(self.words[index], dtype=np.float32)
        tag = self.tags[index]
        word = torch.from_numpy(word).squeeze()
        data = {"word": word, "labels": tag}
        return data


def open_and_split_file(file_path, glove_twitter, vec_num):
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
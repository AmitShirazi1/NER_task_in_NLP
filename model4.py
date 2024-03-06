import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from gensim import downloader
import torch.optim as optim

from lstm_classes import competativeLSTM
from datasets_classes import CompetativeDataset
from lstm_classes import train as train_with_f1

TRAIN_PATH = "/home/student/hw2/NER_task_in_NLP/data/train.tagged"
DEV_PATH = "/home/student/hw2/NER_task_in_NLP/data/dev.tagged"

def represent_words_as_vectors(glove_model, word2vec_model, vec_num, sentences):
    """ Represent words as vectors using the concatenation of 2 different models.
        params: glove_model - the glove model
                word2vec_model - the word2vec model
                vec_num - the total number of features in the word vector
                sentences - the sentences to represent
        return: sentences_vectors - the sentences represented as vectors. """
    words = []
    sentences_vectors = []
    words_per_model = dict()  # A dictionary to hold the words' vector representation for each model
    max_len = max([len(sentence) for sentence in sentences])  # The maximum length of a sentence

    for sentence in sentences:
        for word_representation_model in [glove_model, word2vec_model]:
            # Represent each word in the sentence as a vector according to each model
            words_per_model[word_representation_model] = list()  # A list to hold the words' vector representation for each model
            for w_idx, word in enumerate(sentence):  # Iterate over the words in the sentence
                temp_word = np.array([])
                if (word not in word_representation_model):
                    # If the word is not in the model, represent it as the average of the vectors of its stemmed words and the words in its window
                    stemmed_words = list()
                    # Stemming 
                    for i in range(min(len(word), 5)):
                        if word[i:] in word_representation_model:
                            stemmed_words.append(word_representation_model[word[i:]])
                        for j in range(1, min(len(word), 5)):
                            if word[i:-j] in word_representation_model:
                                stemmed_words.append(word_representation_model[word[i:-j]])
                    average_on_stemmed = np.mean(np.array(stemmed_words), axis=0) if stemmed_words else np.zeros(word_representation_model.vector_size)

                    window_words = list()
                    # Window
                    for i in range(1, 3):
                        if len(words_per_model[word_representation_model]) > i:
                            window_words.append(words_per_model[word_representation_model][-i])
                        if w_idx < len(sentence) - i:
                            if sentence[w_idx + i] in word_representation_model:
                                window_words.append(word_representation_model[sentence[w_idx + i]])
                    average_on_window = np.mean(np.array(window_words), axis=0) if window_words else np.zeros(word_representation_model.vector_size)
            
                    temp_word = np.mean(np.array([average_on_stemmed, average_on_window]), axis=0) # average of both methods
                else:
                    # If the word is in the model, represent it as the vector of the word
                    temp_word = word_representation_model[word]
                
                words_per_model[word_representation_model].append(temp_word)

        for i in range(len(words_per_model[glove_model])):
            # For each word in the sentence, we concatenate the vectors of the word from the 2 models
            words.append(np.concatenate((words_per_model[glove_model][i], words_per_model[word2vec_model][i])))
        
        # save the sentence's vectors
        for i in range(len(words), max_len):
            words.append(np.zeros(vec_num))
        sentences_vectors.append(words)
        words = []
    return sentences_vectors


def open_split_file_and_calc_weights(file_path, glove_model, word2vec_model, vec_num):
    """
    Open a file and split it into sentences and tags.
    Represent the words as vectors using the concatenation of 2 different models (GLoVe and word2vec).
    @param file_path: the path to the file
    @param vec_num: the total number of features in the word vector
    @return: sentences_vectors - list of sentences where each word in sentence is represented as a vector
    @return: sentences_tags - the tags of the words 
    @return: [count_0, count_1] - the count of the 0 and 1 tags in the file
    """
    with open(file_path) as f:
        lines = f.readlines()
        sentences = []
        sentences_tags = []
        words_str = []
        tags = []
        count_0, count_1 = 0, 0
        for line in lines:
            word = line.rstrip()
            if word == '':
                sentences.append(words_str)
                sentences_tags.append(tags)
                count_0 += tags.count(0)
                count_1 += tags.count(1)
                words_str = []
                tags = []
                continue
            try:
                word, tag = word.split("\t")
            except:
                continue
            word = word.lower()
            words_str.append(word)
            tags.append(0 if tag == "O" else 1)
    sentences_vectors = represent_words_as_vectors(glove_model, word2vec_model, vec_num, sentences)
    return np.array(sentences_vectors), sentences_tags, [count_0, count_1]

def open_and_split_test_file(file_path, glove_model, word2vec_model, vec_num):
    """
    Open a test file and split it into sentences and tags.
    Represent the words as vectors using the concatenation of 2 different models (GLoVe and word2vec).
    @param file_path: the path to the file
    @param vec_num: the total number of features in the word vector
    @return: sentences_vectors - list of sentences where each word in sentence is represented as a vector
    @return: sentences_tags - the tags of the words, because it is a test file it is set to 2 (a non-existing label)
    @return: original_sentences - the original sentences 
    """
    with open(file_path) as f:
        lines = f.readlines()
        sentences = []
        original_sentences = list()
        sentences_tags = []
        words_str = []
        original_words_str = list()
        tags = []
        for line in lines:
            word = line.strip()
            if word == '':
                sentences.append(words_str)
                original_sentences.append(original_words_str)
                sentences_tags.append(tags)
                words_str = []
                original_words_str = list()
                tags = []
                continue
            original_words_str.append(word)
            word = word.lower()
            words_str.append(word)
            tags.append(2)
    sentences_vectors = represent_words_as_vectors(glove_model, word2vec_model, vec_num, sentences)
    return np.array(sentences_vectors), sentences_tags, original_sentences

def train_with_test(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    """
    Trains the given model (same as the one in the tutorial).
    Uses the trained model, to predict the labels of dev set.
    Finally calculates and prints f1 score on dev.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)

    lr_reducer = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    for epoch in range(num_epochs):
        model.train()
        accumelative_loss = 0
        for batch in data_loaders['train']:
            batch_size = 0
            for k, v in batch.items():
                batch[k] = v.to(device)
                batch_size = v.shape[0]

            optimizer.zero_grad()
            _, loss = model(**batch)
            loss.backward()  # The important part
            accumelative_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # TODO: Delete when running models 2 and 3.
            optimizer.step()
        lr_reducer.step(accumelative_loss / len(data_loaders['train']))

    # Now run on the test dataset.
    model.eval()
    predictions = torch.tensor([])
    tags = torch.tensor([])
    for batch in data_loaders["test"]:
        batch_size = 0
        for k, v in batch.items():
            batch[k] = v.to(device)
            batch_size = v.shape[0]

        optimizer.zero_grad()    
        with torch.no_grad():
            outputs, _ = model(**batch)
            pred = outputs.argmax(dim=-1).clone().detach().cpu()
            predictions = torch.cat((predictions, pred), 0)
        tags = torch.cat((tags, (batch["labels"].clone().detach().cpu())), 0)
        
        # Filter out padding (where labels are -1)
        flat_predictions = predictions.view(-1)
        flat_tags = tags.view(-1)
        mask = flat_tags != -1
                
    return flat_predictions[mask]


def model4(train_path, dev_path, test_path=None):
    GLOVE_PATH = f'glove-twitter-200'
    glove_model = downloader.load(GLOVE_PATH)
    word2vec_model = downloader.load('word2vec-google-news-300')
    vec_num = 500  # number of features in word vector

    train_words, train_labels, train_tags_frequencies = open_split_file_and_calc_weights(train_path, glove_model, word2vec_model, vec_num)
    padded_train_tags = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in train_labels], batch_first=True, padding_value=-1)
    train_dataset = CompetativeDataset(sentences=train_words, labels=padded_train_tags)

    num_epochs = 15
    learning_rate = 0.001
    num_stacked_layers = 4  # Number of stacked lstm layers, in this model, we do not stack layers.
    num_classes = 2  # Number of output classes

    lstm_model = competativeLSTM(num_classes, vec_num, int(0.9*vec_num), int(vec_num/2), num_stacked_layers, tags_weights=[1, train_tags_frequencies[0]/train_tags_frequencies[1]])  # Initiate the model
    optimizer = AdamW(lstm_model.parameters(), lr=learning_rate)  # Adam optimizer

    print("-----------------Model 4-----------------")
    if not test_path:
        dev_words, dev_labels, _ = open_split_file_and_calc_weights(dev_path, glove_model, word2vec_model, vec_num)
        padded_dev_tags = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in dev_labels], batch_first=True, padding_value=-1)
        dev_dataset = CompetativeDataset(sentences=dev_words, labels=padded_dev_tags)
        datasets = {"train": train_dataset, "dev": dev_dataset}
        predictions = train_with_f1(model=lstm_model, data_sets=datasets, optimizer=optimizer, num_epochs=num_epochs)
    else:
        test_words, test_labels, test_sentences = open_and_split_test_file(test_path, glove_model, word2vec_model, vec_num)
        padded_test_tags = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in test_labels], batch_first=True, padding_value=-1)
        test_dataset = CompetativeDataset(sentences=test_words, labels=padded_test_tags)
        datasets = {"train": train_dataset, "test": test_dataset}
        predictions = train_with_test(model=lstm_model, data_sets=datasets, optimizer=optimizer, num_epochs=num_epochs)
        return test_sentences, predictions, lstm_model
    print("----------------------------------------")


if __name__ == "__main__":
    model4(TRAIN_PATH, DEV_PATH)

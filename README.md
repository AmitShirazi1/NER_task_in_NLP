# Named Entity Recognition (NER) in NLP

## Project Overview

This project implements a Named Entity Recognition (NER) system using various machine learning models, including a Feed-Forward Net and a Bidirectional LSTM. The goal is to accurately classify words in a sentence into categories such as named entities, leveraging different word embedding techniques like GloVe and Word2Vec.

## Project Structure

The project folder structure is as follows:

```
NER_Task_in_NLP/
├── data/
│   ├── train.tagged
│   ├── dev.tagged
│   └── test.untagged
├── FeedForward.py
├── datasets_classes.py
├── generate_comp_tagged.py
├── lstm_classes.py
├── model1.py
├── model2.py
├── model3.py
└── model4.py
```

### Data Files

- `data/train.tagged`: Training data with words and corresponding NER tags.
- `data/dev.tagged`: Development data for model evaluation with words and tags.
- `data/test.untagged`: Test data for final model predictions without known tags.

### Code Files

- **FeedForward.py**: Implements a Feed-Forward Neural Network for NER classification.
  
- **datasets_classes.py**: Contains the `CompetativeDataset` class that prepares datasets for training and testing.

- **generate_comp_tagged.py**: Responsible for generating tagged datasets for evaluation.

- **lstm_classes.py**: Includes the `competativeLSTM` class, which defines the architecture of the competitive LSTM model used for NER.

- **model1.py**

- **model2.py**

- **model3.py**

- **model4.py**

## Models Descriptions

This project includes four models for Named Entity Recognition, each employing different architectures and methodologies. Below is a breakdown of each model and its distinctive features.

### 1. K-Nearest Neighbors (KNN) - `model1.py`

The first model implements a K-Nearest Neighbors algorithm for NER. It uses a vector representation of words obtained through word embeddings. The KNN algorithm classifies entities based on the proximity of word vectors in the feature space, effectively leveraging the distance metric to predict the NER tags.

### 2. Feed-Forward Neural Network (FF) - `model2.py`

The second model is a Feed-Forward Neural Network. This model processes the word embeddings in a single pass without the sequential connections inherent in LSTMs. It consists of fully connected layers that capture the relationships between words, making it suitable for recognizing patterns in the NER task. The model utilizes the GloVe word embeddings for feature representation and trains on the provided tagged dataset.

### 3. Long Short-Term Memory (LSTM) - `model3.py`

The third model employs a Long Short-Term Memory (LSTM) network. LSTMs are a type of recurrent neural network that is particularly effective for sequence prediction tasks. In this model, LSTMs take sequences of word embeddings as input, allowing the model to learn temporal dependencies between words. The implementation includes hyperparameters such as the number of epochs and learning rate, optimizing the LSTM's performance on the NER dataset.

### 4. Competitive LSTM - `model4.py`

The fourth model features a competitive LSTM approach, which is an advanced version of the standard LSTM model. This model concatenates word representations from two different embedding models: GloVe and Word2Vec. It uses a multi-layer LSTM architecture to capture complex dependencies within the input sequences. The model also integrates tag frequency weights to handle class imbalance in the training data, improving the accuracy of entity recognition across different classes.

## Running the Models

To run any of the models, execute the respective Python file. Ensure that the required libraries (PyTorch, Gensim, NumPy) are installed in your environment.

### Example Usage

```bash
python model1.py
python model2.py
python model3.py
python model4.py
```

## Dependencies

- PyTorch
- Gensim
- NumPy

## Description of Key Functions

### `represent_words_as_vectors(glove_model, word2vec_model, vec_num, sentences)`

This function represents words as vectors using the concatenation of GloVe and Word2Vec embeddings. It handles words not found in the embeddings by averaging vectors of their stemmed forms and context.

### `open_split_file_and_calc_weights(file_path, glove_model, word2vec_model, vec_num)`

This function opens the specified file, splits it into sentences and tags, and converts words into vector representations.

### `open_and_split_test_file(file_path, glove_model, word2vec_model, vec_num)`

This function processes the test file and prepares sentences and tags for evaluation.

### `train_with_test(model, data_sets, optimizer, num_epochs: int, batch_size=16)`

Trains the model on the training dataset and evaluates it on the test dataset, returning the predictions.

### `model4(train_path, dev_path, test_path=None)`

This function initializes the GloVe and Word2Vec models, prepares datasets, and trains the competitive LSTM model. It also handles evaluation based on whether a test dataset is provided.

## Conclusion

This project demonstrates the implementation of various machine learning models for Named Entity Recognition. By comparing different algorithms, the goal is to identify the strengths and weaknesses of each approach, ultimately aiming to enhance entity recognition accuracy in natural language processing tasks.

## Acknowledgments

- GloVe and Word2Vec models for providing pre-trained word embeddings.
- The PyTorch framework for enabling deep learning model development.

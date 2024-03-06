from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

class competativeLSTM(nn.Module):
    """ This class is a model for the LSTM network. """
    def __init__(self, num_classes, input_size, hidden_size, hidden2_size, num_stacked_layers, dropout_prob=0.2, batch_size=16, tags_weights=[1.0, 10.0]):
        super(competativeLSTM, self).__init__()
        self.num_classes = num_classes  # We have 2 classes, binary.
        self.input_size = input_size  # The number of expected features in the input x.
        self.hidden_size = hidden_size  # number of features in hidden state.
        self.num_stacked_layers = num_stacked_layers  # number of stacked lstm layers.

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=self.num_stacked_layers, bidirectional=True, dropout=dropout_prob)  # Bidirectional LSTM
        self.layer1 =  nn.Linear(2 * hidden_size, hidden2_size)  # linear layer 1 
        self.layer2 = nn.Linear(hidden2_size, num_classes)  # linear layer 2 

        self.relu = nn.LeakyReLU(0.01) # activation function
        self.sigmoid = nn.Sigmoid() # activation function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device (cuda if cuda availiable)
        # cross entropy loss: ignores the label -1 because it represents the padding of our sentences, 
        # and adds a weight for each other label to combat class imbalnce
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(tags_weights).to(self.device), ignore_index=-1) 
        self.dropout = nn.Dropout(dropout_prob) # dropout

    
    def forward(self, sentence, labels=None):
        sentence = sentence.type(torch.FloatTensor).to(self.device)
        num_layers = self.num_stacked_layers * 2
        h_0 = Variable(torch.zeros(num_layers, sentence.size(0), self.hidden_size)).to(self.device)  # Short term memory.
        c_0 = Variable(torch.zeros(num_layers, sentence.size(0), self.hidden_size)).to(self.device)  # Long term memory.
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(sentence, (h_0, c_0))  # Perform lstm with relation to input, hidden, and internal state

        # Reshape output for linear layer.
        # output shape: [batch_size=16, max_len=41, 2 * hidden_size]
        batch_size, seq_length, features = output.shape
        output = output.contiguous().view(batch_size * seq_length, features)

        out = self.relu(output)
        out = self.layer1(out)  # First Dense
        out = self.relu(out)  # Activation function - Relu
        out = self.dropout(out)
        out = self.layer2(out)  # Second layer
        out = self.sigmoid(out) # Activation function - Softmax
        out = out.view(batch_size, seq_length, -1)
        if torch.tensor(2) in labels:
            return out, None
        labels = labels.view(-1)
        loss = self.loss(out.view(-1, out.shape[-1]), labels)  # Reshape output to [batch_size * seq_length, num_classes]
        return out, loss

''' This code was inspired by the following source: https://cnvrg.io/pytorch-lstm/
    Because we thought it was more suitable for our uses than the one studied in class. '''
class model3_LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, hidden2_size, num_stacked_layers):
        super(model3_LSTM, self).__init__()
        self.num_classes = num_classes  # We have 2 classes, binary.
        self.input_size = input_size  # The number of expected features in the input x.
        self.hidden_size = hidden_size  # number of features in hidden state.
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)  # lstm
        self.layer1 =  nn.Linear(hidden_size, hidden2_size)  # Layer 1 in the LSTM
        self.layer2 = nn.Linear(hidden2_size, num_classes)  # Layer 2 in the LSTM

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss = nn.CrossEntropyLoss()

    
    def forward(self, word, labels=None):
        word = word.unsqueeze(1)
        h_0 = Variable(torch.zeros(self.num_stacked_layers, word.size(0), self.hidden_size)).to(self.device)  # Short term memory.
        c_0 = Variable(torch.zeros(self.num_stacked_layers, word.size(0), self.hidden_size)).to(self.device)  # Long term memory.
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(word, (h_0, c_0))  # Perform lstm with relation to input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # Reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.layer1(out)  # First Dense
        out = self.relu(out)  # Activation function - Relu
        out = self.layer2(out)  # Second layer
        out = self.softmax(out) # Activation function - Softmax
        # pred = outputs.argmax(dim=-1).clone().detach().cpu()
        if labels is None:
            return out, None
        loss = self.loss(out, labels)
        return out, loss

def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    """
    Trains the given model (same as the one in the tutorial).
    Uses the trained model, to predict the labels of dev set.
    Finally calculates and prints f1 score on dev.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "dev": DataLoader(data_sets["dev"], batch_size=batch_size, shuffle=False)}
    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        for batch in data_loaders['train']:
            batch_size = 0
            for k, v in batch.items():
                batch[k] = v.to(device)
                batch_size = v.shape[0]

            optimizer.zero_grad()
            _, loss = model(**batch)
            loss.backward()  # The important part
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Prevents the exploding gradient problem
            optimizer.step()
                
    # Now use the dev dataset to evaluate the model.
    for evaluate in ["train", "dev"]:
        model.eval()
        predictions = torch.tensor([])
        tags = torch.tensor([])
        for batch in data_loaders[evaluate]:
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
        score = f1_score(flat_tags[mask], flat_predictions[mask])
        print(f'F1 score on {evaluate}: {score}')
                
    return flat_predictions[mask]
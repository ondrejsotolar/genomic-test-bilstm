import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn, GenomicClfDataset
from genomic_benchmarks.models.torch import CNN
from genomic_benchmarks.dataset_getters.utils import coll_factory, LetterTokenizer, build_vocab
from genomic_benchmarks.data_check import info
from torchtext.vocab import Vocab
from sklearn.metrics import f1_score
from genomic_benchmarks.data_check.info import labels_in_order


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(BiLSTM, self).__init__()
        self.number_of_output_neurons = 1
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.output_activation = nn.Sigmoid()

        self.hidden_size = 64
        drp = 0.1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)

        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        out_act = self.output_activation(out)
        return out_act

    def train_loop(self, dataloader, optimizer):
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = self(x)
            loss = self.loss(pred, y)
            loss.backward()
            optimizer.step()

        #       train acc
        # todo: optimize counting of acc
        size = dataloader.dataset.__len__()
        num_batches = len(dataloader)
        train_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                train_loss += self.loss(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()

        train_loss /= num_batches
        correct /= size
        print(f"Train metrics: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    def train(self, dataloader, epochs):
        optimizer = torch.optim.Adam(self.parameters())
        for t in range(epochs):
            print(f"Epoch {t}")
            self.train_loop(dataloader, optimizer)


def run():
    # Load dataset and print info
    train_dset: GenomicClfDataset = HumanEnhancersCohn('train', version=0)
    info("human_enhancers_cohn", 0)

    tokenizer = get_tokenizer(LetterTokenizer())
    vocabulary: Vocab = build_vocab(train_dset, tokenizer, use_padding=False)

    print("vocab len:", vocabulary.__len__())
    print(vocabulary.get_stoi())

    collate = coll_factory(vocabulary, tokenizer, _device, pad_to_length=None)
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, collate_fn=collate)

    # Init
    # model = CNN(
    #     number_of_classes=2,
    #     vocab_size=vocabulary.__len__(),
    #     embedding_dim=100,
    #     input_len=500
    # ).to(_device)
    model = BiLSTM(
        vocab_size=vocabulary.__len__(),
        embedding_dim=100
    ).to(_device)

    model.train(train_loader, epochs=5)

    test_dset = HumanEnhancersCohn('test', version=0)
    test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, collate_fn=collate)

    # Test
    predictions = []
    for x, y in test_loader:
        output = torch.round(model(x))
        for prediction in output.tolist():
            predictions.append(prediction[0])

    labels = labels_in_order(dset_name='human_enhancers_cohn')
    print('F1: {0:.2f}'.format(f1_score(labels, predictions)))


if __name__ == '__main__':
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(_device))
    run()

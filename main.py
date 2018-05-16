import io
import gzip
import os
import tempfile

from deepsense import neptune
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import log_loss, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

ctx = neptune.Context()

EPOCH_NR = ctx.params.epoch_nr
DENSE_UNITS = ctx.params.dense_units
BATCH_SIZE = ctx.params.batch_size
LEARNING_RATE = ctx.params.learning_rate
INPUT_SHAPE = (1, 28, 28)
CLASSES = 10


def load_data():
    train_dataset = FashionMNIST(root='./cache', download=True, train=True, transform=ToTensor())
    eval_dataset = FashionMNIST(root='./cache', download=False, train=False, transform=ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, eval_loader


def get_model():
    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.flat_features_nr = self._flat_features_nr(self.features)

            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(self.flat_features_nr, DENSE_UNITS),
                nn.Dropout(p=0.2),
                nn.Linear(DENSE_UNITS, CLASSES),
                nn.LogSoftmax()
            )

        def _flat_features_nr(self, features):
            dummy_input = Variable(torch.ones(1, *INPUT_SHAPE))
            f = features(dummy_input)
            return int(np.prod(f.size()[1:]))

        def forward(self, x):
            features = self.features(x)
            flat_features = features.view(-1, self.flat_features_nr)
            prediction = self.classifier(flat_features)
            return prediction

    return Classifier()


def train(model, optimizer, criterion, batch_generator_train, batch_generator_eval):
    if torch.cuda.is_available():
        model.cuda()

    for epoch_id in range(EPOCH_NR):
        for batch_id, batch_data in enumerate(batch_generator_train):
            X_batch, y_batch = Variable(batch_data[0]), Variable(batch_data[1])

            if torch.cuda.is_available():
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            optimizer.zero_grad()
            y_batch_pred = model(X_batch)
            loss_batch = criterion(y_batch_pred, y_batch)
            loss_batch.backward()
            optimizer.step()

            loss_value = loss_batch.data.cpu().numpy()[0]
            batch_nr = epoch_id * len(batch_generator_train) + batch_id
            batch_msg = 'Batch {} log-loss {}'.format(batch_nr, loss_value)
            print(batch_msg)

            ctx.channel_send('Batch log-loss', batch_nr, loss_value)

            if batch_id == 10:  # len(batch_generator_train):
                break

        log_loss, accuracy, histograms = score_model(model, batch_generator_eval)

        print(histograms)

        epoch_msg_log_loss = 'Epoch {} validation log-loss {}'.format(epoch_id, log_loss)
        epoch_msg_accuracy = 'Epoch {} validation accuracy {}'.format(epoch_id, accuracy)
        print(epoch_msg_log_loss)
        print(epoch_msg_accuracy)

        ctx.channel_send('Epoch validation log-loss', epoch_id, log_loss)
        ctx.channel_send('Epoch validation accuracy', epoch_id, accuracy)


def score_model(model, batch_generator):
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    y_pred, y_true = [], []
    for batch_id, batch_data in enumerate(batch_generator):
        X_batch = Variable(batch_data[0], volatile=True)
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()

        y_batch_pred = model(X_batch)
        y_pred.append(y_batch_pred.data.cpu().numpy())
        y_true.append(batch_data[1].numpy())

        if batch_id == 10:  # len(batch_generator):
            break

    model.train()

    y_true = np.concatenate(y_true)
    y_pred = np.vstack(y_pred)

    log_loss, accuracy = get_scores(y_true, y_pred)
    prediction_histogram = get_histograms(y_true, y_pred)
    return log_loss, accuracy, prediction_histogram


def get_scores(y_true, y_pred):
    return log_loss(y_true, y_pred, labels=list(range(CLASSES))), accuracy_score(y_true, np.argmax(y_pred, axis=1))


def get_histograms(y_true, y_pred):
    y_true_counts = pd.DataFrame(y_true)[0].value_counts().sort_index()
    y_pred_counts = pd.DataFrame(np.argmax(y_pred, axis=1))[0].value_counts()
    for index in y_true_counts.index:
        y_pred_counts[index] = y_pred_counts.get(index, 0)
    y_pred_counts.sort_index(inplace=True)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    y_true_counts.plot(ax=ax1, kind='bar')
    y_pred_counts.plot(ax=ax2, kind='bar')

    with tempfile.NamedTemporaryFile() as filepath:
        plt.savefig(filepath)
        image_pil = Image.read(filepath, 'r')
    return image_pil


def main():
    train_data, eval_data = load_data()
    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = F.nll_loss

    train(model, optimizer, criterion, train_data, eval_data)


if __name__ == "__main__":
    main()

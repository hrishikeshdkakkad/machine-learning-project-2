'''
Comparing single layer MLP with deep MLP (using PyTorch)
'''

import torch
from scipy.io import loadmat
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron(no_input):
    class Net(nn.Module):
        def __init__(self, inp):
            super().__init__()

            # Network Parameters
            n_hidden_1 = 1024  # 1st layer number of features
            n_hidden_2 = 256  # 2nd layer number of features
            n_input = inp  # data input
            n_classes = 10

            # Initialize network layers
            self.layer_1 = nn.Linear(n_input, n_hidden_1)
            self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
            self.out_layer = nn.Linear(n_hidden_2, n_classes)

        def forward(self, x):
            x = F.relu(self.layer_1(x))
            x = F.relu(self.layer_2(x))
            x = self.out_layer(x)
            return x

    return Net(no_input)


# Do not change this
def preprocess():
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i in range(10):
        train_i = mat['train' + str(i)]
        test_i = mat['test' + str(i)]
        for ex in train_i:
            train_data.append(ex)
            train_label.append(i)
        for test_ex in test_i:
            test_data.append(test_ex)
            test_label.append(i)

    train_data = np.array(train_data) / 255
    # train_data = train_data[:, ~np.all(train_data[1:] == train_data[:-1], axis=0)]
    test_data = np.array(test_data) / 255
    # test_data = test_data[:, ~np.all(test_data[1:] == test_data[:-1], axis=0)]
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    train_label = np.reshape(train_label, (len(train_label), 1))
    test_label = np.reshape(test_label, (len(test_label), 1))

    train_with_label = np.hstack([train_data, train_label])
    test_with_label = np.hstack([test_data, test_label])

    np.random.shuffle(train_with_label)
    np.random.shuffle(test_with_label)

    validation_data = train_with_label[-10000:, :-1]
    validation_label = train_with_label[-10000:, -1]

    train_data = train_with_label[:50000, :-1]
    train_label = train_with_label[:50000, -1]

    class dataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

        def get_no_of_features(self):
            return self.X.shape[1]

    trainset = dataset(train_data, train_label.astype('int8'))
    validset = dataset(validation_data, validation_label.astype('int8'))
    testset = dataset(test_data, test_label.astype('int8'))

    return trainset, validset, testset


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y.type(torch.LongTensor))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.type(torch.LongTensor).flatten()
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Parameters
learning_rate = 0.05
training_epochs = 50
batch_size = 150

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# load data
trainset, validset, testset = preprocess()
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Construct model
model = create_multilayer_perceptron(trainset.get_no_of_features()).to(device)

# Define loss and openptimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training cycle
for t in range(training_epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, cost, optimizer)
print("Optimization Finished!")
test(test_dataloader, model, cost)

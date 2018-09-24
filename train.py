import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt

from dataset import Dataset
from mlp import MLP


def load_data(data_directory):
	train_df = pd.read_csv(os.path.join(data_directory, 'train.csv'), header=None)
	test_df = pd.read_csv(os.path.join(data_directory, 'test.csv'), header=None)
	return train_df, test_df

def partition_data(df, split=.2):
	mask = np.random.rand(df.shape[0]) < split
	return df[~mask], df[mask]

def run_epoch(generator, epoch_size, optimizer=None):
	losses = []
	batch_count = 0
	for x_batch, y_batch in generator:
		logits = mlp(x_batch)
		loss = loss_function(logits, y_batch)
		if optimizer:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		losses.append(loss.cpu().detach().numpy())
		batch_count += 1
		if batch_count >= epoch_size:
			break
	return np.array(losses).mean()

data_params = {
		'batch_size': 8,
		'shuffle': True}

# data_params = {
# 		'batch_size': 8,
# 		'shuffle': True,
# 		'num_workers': 4}

num_epochs = 100
train_epoch_size = 500
validate_epoch_size = 100
learning_rate = 1e-4

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

train_df, test_df = load_data('data/poker_hand')
train_df, validation_df = partition_data(train_df)
test_df = test_df.dropna()

train_set = Dataset(train_df, device)
validation_set = Dataset(validation_df, device)
test_set = Dataset(test_df, device)

train_generator = data.DataLoader(train_set, **data_params)
validation_generator = data.DataLoader(validation_set, **data_params)
test_generator = data.DataLoader(test_set, **data_params)

mlp = MLP(
		input_dimension=train_df.shape[1] - 1,
		output_dimension=train_df[train_df.columns[-1]].unique().shape[0],
		num_layers=2)

mlp = mlp.to(device)

loss_function = nn.modules.loss.CrossEntropyLoss()

optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

train_losses = []
validation_losses = []

start_time = time.time()
for epoch in range(num_epochs):
	print('Epoch: {}'.format(epoch))
	print('Runtime: {}'.format(time.time() - start_time))
	train_losses.append(run_epoch(train_generator, train_epoch_size, optimizer))
	validation_losses.append(run_epoch(validation_generator, validate_epoch_size))
	torch.save(mlp, 'weights/mlp_weights_epoch_{}.pth'.format(epoch))

plt.title('Losses')
plt.plot(train_losses, '-r', label='train')
plt.plot(validation_losses, '-b', label='validation')
plt.legend(loc='upper right')
plt.savefig('losses.png')
plt.show()

accuracy_count = 0
for x_batch, y_batch in test_generator:
	probabilities = mlp(x_batch, return_logits=False)
	indices = torch.multinomial(probabilities, 1)
	accuracy_count += (y_batch.view(-1) == indices.view(-1)).sum()
print('Test Accuracy: {}'.format(float(accuracy_count) / test_df.shape[0]))
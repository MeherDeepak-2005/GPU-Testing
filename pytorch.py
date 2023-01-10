from datetime import datetime
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


Nclass = 500
x1 = np.random.random((Nclass, 2)) + np.array([0, -2])
x2 = np.random.random((Nclass, 2)) + np.array([2, 2])
x3 = np.random.random((Nclass, 2)) + np.array([-2, 2])
X = np.vstack([x1, x2, x3])

y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N = len(y)
labels = np.zeros((N, 3))
for i in range(N):
	labels[i, y[i]] = 1


class NeuralNetwork(nn.Module):
	def __init__(self, lr, device):
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(2, 128)
		self.fc2 = nn.Linear(128, 3)

		self.optimizer = optim.SGD(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()

		self.device = T.device(device)
		self.to(self.device)

	def forward(self, x):
		y = F.sigmoid(self.fc1(x))
		y = F.softmax(self.fc2(y))
		return y


X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.33, shuffle=True)


def test_worker(device):
	global Y_train, X_train, Y_test, X_test
	batch_size = 20
	nn = NeuralNetwork(lr=0.0001, device=device)

	X_train, X_test, Y_train, Y_test = T.tensor(X_train, dtype=T.float).to(device), T.tensor(X_test, dtype=T.float32).to(device), T.tensor(
		Y_train, dtype=T.float32).to(device), T.tensor(Y_test, dtype=T.float32).to(device)

	epochs = 1000
	for epoch in range(epochs):
		cost_of_epoch = 0
		for i in range(0, len(X_train), batch_size):
			nn.optimizer.zero_grad()
			x = X_train[i: i+batch_size]
			y_target = Y_train[i:i+batch_size]
			y_pred = nn(x=x)
			cost = nn.loss(y_pred, y_target)
			cost_of_epoch += cost.item()
			cost.backward()
			nn.optimizer.step()

		print("Epoch", epoch, 'training loss -', cost_of_epoch)


gpu_start_time = datetime.now()
test_worker('mps')
GPU_time = (datetime.now() - gpu_start_time).seconds
print("GPU testing completed within -", GPU_time)

cpu_start_time = datetime.now()
test_worker('cpu')
CPU_Time = (datetime.now() - cpu_start_time).seconds
print("CPU testing completed within - ", CPU_Time)

if CPU_Time > GPU_time:
	print("GPU WON by ", (CPU_Time - GPU_time), "seconds")
else:
	print("CPU WON by ", (GPU_time - CPU_Time), "seconds")

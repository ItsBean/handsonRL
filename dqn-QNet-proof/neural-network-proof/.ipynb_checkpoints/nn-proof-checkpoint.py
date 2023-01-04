import torch
import torch.nn as nn


# Define the model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Create an instance of the model
model = Net(input_size=2, hidden_size=10, output_size=1)

# Define the optimization algorithm and the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Generate some synthetic data
x_train = torch.randn(1000, 2)
y_train = x_train[:, 0] ** 2 + x_train[:, 1] ** 2

# Train the model
for epoch in range(1000):
    # Forward pass
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update the weights
    optimizer.step()

# do some test on the model
x_test = torch.randn(1000, 2)
y_test = x_test[:, 0] ** 2 + x_test[:, 1] ** 2
y_pred = model(x_test)
# print the difference:
diff = []
index = 0

# ytest and ypred to list
ytest = y_test.tolist()
ypred = y_pred.tolist()

# calculate difference:
for index in range(len(ytest)):
    diff.append(abs(ypred[index][0] - ytest[index]))

# get avg of diff
avg = sum(diff) / len(diff)
print(avg, diff, sep="\n")

#
# for element in y_pred_tolist:
#     diff.append(element - y_test_tolist[index])
#     index += 1
# print(diff)
# plot y_pred given x_test and y_test given x_test

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and a 3D Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make predictions using the model
y_pred = model(x_train).flatten()
# convert x_train and y_pred to list
x_train_tolist = np.array(x_train.tolist())
y_pred = np.array(y_pred.tolist())
# Create a scatter plot
ax.scatter(x_train[:, 0], x_train[:, 1], y_pred.squeeze())

# Add axis labels and a title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('Model predictions')

plt.show()

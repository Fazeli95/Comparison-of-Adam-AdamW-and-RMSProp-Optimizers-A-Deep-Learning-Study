import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc4.bias, 0)

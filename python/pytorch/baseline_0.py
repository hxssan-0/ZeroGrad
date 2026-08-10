import torch
import torch.nn as nn

cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, stride=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(16 * 1 * 1, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1)

images = torch.randn(64, 1, 28, 28)
labels = torch.randint(0, 10, (64,))

for _ in range(3):
    optimizer.zero_grad()
    loss = criterion(cnn(images), labels)
    loss.backward()
    optimizer.step()
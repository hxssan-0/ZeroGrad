import _zerograd_backend as zg
import time

train_data = zg.load_mnist(
    "../../datasets/mnist/train_images.idx3-ubyte",
    "../../datasets/mnist/train_labels.idx1-ubyte"
)

test_data = zg.load_mnist(
    "../../datasets/mnist/test_images.idx3-ubyte",
    "../../datasets/mnist/test_labels.idx1-ubyte"
)

mlp = zg.Sequential()
mlp.add(zg.Linear(784, 128))
mlp.add(zg.ReLU())
mlp.add(zg.Linear(128, 10))

batch_size = 64
train_loader = zg.DataLoader(train_data, batch_size)

lr = 1e-1
optimizer = zg.Optimizer(mlp.parameters(), lr)

print("====== BEGINNING TRAINING LOOP ======")
NUM_EPOCHS = 15

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    print(f"========= EPOCH #{epoch+1} =============")
    train_loader.shuffle()
    train_loader.reset()
    num_batches = 0

    while train_loader.has_next():
        images, labels = train_loader.next_batch()
        optimizer.zero_grad()
        logits = mlp.forward(images)
        loss = zg.ce_loss(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]

        num_batches += 1

        if num_batches % 100 == 0:
            print(f"Batch #{num_batches} loss: {loss.data[0]}")

    print(f"Avg. loss: {running_loss / num_batches}")
    print("================================================")

print("====== BEGINNING TESTING ======")

test_loader = zg.DataLoader(test_data, batch_size)
total_correct = 0
total_samples = 0

while test_loader.has_next():
    images, labels = test_loader.next_batch()
    logits = mlp.forward(images)
    total_correct += zg.count_correct(logits, labels)
    total_samples += len(labels)

test_accuracy = (total_correct / total_samples) * 100.0

print(f"Test Accuracy: {test_accuracy:.2f}% ({total_correct}/{total_samples})")
print("=================================")
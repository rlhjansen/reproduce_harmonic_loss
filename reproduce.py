import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define fully connected network
class FCNN(nn.Module):
    def __init__(self, use_harmonic_loss=False, n=1):
        super(FCNN, self).__init__()
        self.fc = nn.Linear(28*28, 10)
        self.use_harmonic_loss = use_harmonic_loss
        self.n = np.sqrt(10).item()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.use_harmonic_loss:
            d = torch.cdist(x, self.fc.weight, p=2)
            print(f"{x.shape=}, {d.shape=}", end="\r")
            logits = (1 / (d ** self.n))/(1 / (d ** self.n).sum(dim=1, keepdim=True))  # Harmonic probability
        else:
            logits = self.fc(x)
        return logits

# Training function
def train_model(model, loss_fn, optimizer, num_epochs=10, save_path="model.pth"):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(trainloader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(outputs, labels)
            else:
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Define harmonic loss function
def harmonic_loss(logits, labels):
    loss = -torch.log(logits[range(len(labels)), labels])  # Negative log likelihood
    return loss.mean()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for saving models
ce_model_path = "cross_entropy_model.pth"
hl_model_path = "harmonic_loss_model.pth"

# Train or load Cross-Entropy model
model_ce = FCNN(use_harmonic_loss=False).to(device)
if os.path.exists(ce_model_path):
    model_ce.load_state_dict(torch.load(ce_model_path))
    print(f"Loaded Cross-Entropy model from {ce_model_path}")
else:
    optimizer_ce = optim.Adam(model_ce.parameters(), lr=0.001)
    train_model(model_ce, nn.CrossEntropyLoss(), optimizer_ce, save_path=ce_model_path)

# Train or load Harmonic Loss model
model_hl = FCNN(use_harmonic_loss=True, n=1).to(device)
if os.path.exists(hl_model_path):
    model_hl.load_state_dict(torch.load(hl_model_path))
    print(f"Loaded Harmonic Loss model from {hl_model_path}")
else:
    optimizer_hl = optim.Adam(model_hl.parameters(), lr=0.001)
    train_model(model_hl, lambda logits, labels: harmonic_loss(logits, labels), optimizer_hl, save_path=hl_model_path)

# Visualization function
def plot_weights(model, title):
    weights = model.fc.weight.data.cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.show()

# Plot learned weights
plot_weights(model_ce, "Cross-Entropy Loss Weights")
plot_weights(model_hl, "Harmonic Loss Weights")

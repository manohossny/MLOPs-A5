import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import mlflow


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)


def train_and_evaluate(epochs, force_low_accuracy):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    if force_low_accuracy:
        train_dataset = Subset(train_dataset, range(100))
        epochs = 1

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_experiment("Assignment3_ManoHosny")

    with mlflow.start_run() as run:
        mlflow.set_tag("student_id", "202200066")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", 0.001)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("force_low_accuracy", force_low_accuracy)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        mlflow.log_metric("accuracy", accuracy)
        print(f"Test Accuracy: {accuracy:.4f}")

        mlflow.pytorch.log_model(model, "model")

        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID: {run.info.run_id}")
        print(f"Model info saved to model_info.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--force-low-accuracy", action="store_true")
    args = parser.parse_args()

    train_and_evaluate(args.epochs, args.force_low_accuracy)

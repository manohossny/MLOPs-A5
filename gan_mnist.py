import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="GAN MNIST Training with MLflow")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
args = parser.parse_args()

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
latent_dim = 100

# --- DATA LOADING ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- MODELS ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# --- MLFLOW SETUP ---
mlflow.set_experiment("Assignment3_ManoHosny")

with mlflow.start_run(run_name=f"lr={lr}_bs={batch_size}_ep={epochs}"):
    # Log hyperparameters
    mlflow.log_params({
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "Adam",
        "latent_dim": latent_dim,
    })

    # Set tags
    mlflow.set_tag("student_id", "202200066")
    mlflow.set_tag("model_type", "GAN")

    # --- INIT ---
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr)

    # --- TRAINING ---
    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        epoch_acc_real = 0
        epoch_acc_fake = 0
        total_batches = 0

        for i, (real_imgs, _) in enumerate(loader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # 1. Train Discriminator
            opt_d.zero_grad()

            real_labels = torch.ones(b_size, 1).to(device)
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, real_labels)

            predictions_real = (output_real > 0.5).float()
            acc_real = (predictions_real == real_labels).float().mean()

            noise = torch.randn(b_size, latent_dim).to(device)
            fake_imgs = generator(noise)
            fake_labels = torch.zeros(b_size, 1).to(device)
            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, fake_labels)

            predictions_fake = (output_fake < 0.5).float()
            acc_fake = (predictions_fake == fake_labels).float().mean()

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # 2. Train Generator
            opt_g.zero_grad()
            output_gen = discriminator(fake_imgs)
            loss_g = criterion(output_gen, real_labels)
            loss_g.backward()
            opt_g.step()

            epoch_acc_real += acc_real.item()
            epoch_acc_fake += acc_fake.item()
            total_batches += 1

        avg_acc_real = epoch_acc_real / total_batches
        avg_acc_fake = epoch_acc_fake / total_batches
        total_acc = (avg_acc_real + avg_acc_fake) / 2

        # Log metrics per epoch
        mlflow.log_metric("d_loss", loss_d.item(), step=epoch)
        mlflow.log_metric("g_loss", loss_g.item(), step=epoch)
        mlflow.log_metric("d_accuracy", total_acc, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f} | D Accuracy: {total_acc:.4f}")

    # --- GENERATE SAMPLE ---
    z = torch.randn(1, latent_dim).to(device)
    gen_img = generator(z).cpu().detach().squeeze().numpy()
    plt.imshow(gen_img, cmap='gray')
    plt.axis('off')
    plt.savefig('result.png')

    # --- LOG FINAL METRICS AND ARTIFACTS ---
    mlflow.log_metric("final_accuracy", total_acc)
    mlflow.pytorch.log_model(generator, "generator_model")
    mlflow.pytorch.log_model(discriminator, "discriminator_model")
    mlflow.log_artifact("result.png")

    print(f"Final Accuracy: {total_acc:.4f}")
    print("MLflow run logged successfully!")

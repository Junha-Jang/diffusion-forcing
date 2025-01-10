import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import lightning as L
from cnn import CnnEncoder, CnnDecoder

import torch.utils.data as data

torch.set_float32_matmul_precision("high")

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), 3, 64, 64)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), 3, 64, 64)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        x = x.view(x.size(0), 3, 64, 64)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

print("Using", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
print("PyTorch version:", torch.__version__)

# Load data sets
transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
])

train_set = MNIST(os.getcwd(), download=True, train=True, transform=transforms)
test_set = MNIST(os.getcwd(), download=True, train=False, transform=transforms)

# Reduce the size of the datasets
train_set = Subset(train_set, list(range(256)))  # Use only the first 256 samples
test_set = Subset(test_set, list(range(64)))  # Use only the first 64 samples

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)  # Increase batch size
valid_loader = DataLoader(valid_set, batch_size=64)

encoder = CnnEncoder(embedding_size=128)
decoder = CnnDecoder(embedding_size=128)
model = LitAutoEncoder(encoder, decoder)

print("Training model...")

# train with both splits
trainer = L.Trainer(max_epochs=5, enable_progress_bar=True, log_every_n_steps=10)
trainer.fit(model, train_loader, valid_loader)

print("Testing model...")

# test the model
trainer.test(model, dataloaders=DataLoader(test_set, batch_size=64))

print("Done!")
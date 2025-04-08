# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pathlib import Path
from scipy.ndimage import rotate, gaussian_filter

def load_and_transform_mnist(n_samples=5000, rotations=True, thicknesses=True):
    """Load MNIST digits and apply controlled transformations."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Download MNIST dataset
    mnist = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    
    # Select a subset of samples
    indices = np.random.choice(len(mnist), size=n_samples, replace=False)
    
    images = []
    thickness_values = []
    rotation_values = []
    digit_labels = []
    
    for idx in indices:
        img, label = mnist[idx]
        img = img.squeeze().numpy()  # Convert to numpy array
        if label != 1:
            continue
        # Apply random rotation if enabled
        if rotations:
            angle = np.random.uniform(-90, 90)
            img = rotate(img, angle, reshape=False, order=1, mode='constant', cval=0)
            rotation_values.append(angle)
        else:
            rotation_values.append(0)
        
        # Apply random thickness variation if enabled
        if thicknesses:
            # Higher sigma = more blur = thinner lines
            # Lower sigma = less blur = thicker lines
            sigma = np.random.uniform(0.1, 2.2)
            thickness = 1/sigma  # Invert so higher value = thicker
            img = gaussian_filter(img, sigma)
            
            # Normalize and threshold to maintain binary-like appearance
            img = (img > 0.3).astype(float)
            thickness_values.append(thickness)
        else:
            thickness_values.append(1.0)
        
        images.append(img.reshape(-1))  # Flatten to 1D
        digit_labels.append(label)
    
    X = np.vstack(images)
    return (torch.tensor(X, dtype=torch.float32), 
            np.array(rotation_values), 
            np.array(thickness_values),
            np.array(digit_labels))

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2, hidden_dim=256):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output in [0,1] range for images
        )
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def vae_loss(x_hat, x, mu, logvar, kl_weight):
    # Binary cross entropy loss for binary image data
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return recon_loss + kl_weight * kl_div, recon_loss, kl_div

def train_vae(model, data, batch_size=128, n_epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # KL annealing
    def get_kl_weight(epoch, start=0.0, end=1.0, steps=n_epochs//2):
        if epoch < steps:
            # Smooth transition using cubic function
            progress = epoch / steps
            weight = start + (end - start) * (3 * progress**2 - 2 * progress**3)
            return weight
        return end
    
    losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            x_hat, mu, logvar = model(x)
            
            # Compute loss with current KL weight
            kl_weight = get_kl_weight(epoch)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, kl_weight)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
        
        # Track losses
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon / len(dataloader)
        avg_kl = epoch_kl / len(dataloader)
        losses.append((avg_loss, avg_recon, avg_kl))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | KL weight: {kl_weight:.3f}")
    
    return losses

def plot_reconstructions(model, data, n_samples=5):
    model.eval()
    
    # Get random indices
    indices = np.random.choice(len(data), size=n_samples, replace=False)
    
    with torch.no_grad():
        # Get original images
        originals = data[indices]
        
        # Get reconstructions
        recons, _, _ = model(originals)
        
        # Convert to numpy for plotting
        originals = originals.cpu().numpy()
        recons = recons.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstruction
        axes[1, i].imshow(recons[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')
    
    plt.tight_layout()
    plt.show()

def plot_latent_space(z, rotation_values, thickness_values, labels=None):
    """Plot latent space colored by rotation, thickness, and optionally digit labels."""
    fig, axes = plt.subplots(1, 3 if labels is not None else 2, figsize=(18, 6))
    
    # Plot by rotation
    sc1 = axes[0].scatter(z[:, 0], z[:, 1], c=rotation_values, cmap='coolwarm', alpha=0.7)
    axes[0].set_xlabel('Latent Dim 1')
    axes[0].set_ylabel('Latent Dim 2')
    axes[0].set_title('Latent Space: Rotation')
    plt.colorbar(sc1, ax=axes[0], label='Rotation Angle (-45° to 45°)')
    
    # Plot by thickness
    sc2 = axes[1].scatter(z[:, 0], z[:, 1], c=thickness_values, cmap='viridis', alpha=0.7)
    axes[1].set_xlabel('Latent Dim 1')
    axes[1].set_ylabel('Latent Dim 2')
    axes[1].set_title('Latent Space: Thickness')
    plt.colorbar(sc2, ax=axes[1], label='Thickness (thinner to thicker)')
    
    # Plot by digit class if provided
    if labels is not None:
        sc3 = axes[2].scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.7)
        axes[2].set_xlabel('Latent Dim 1')
        axes[2].set_ylabel('Latent Dim 2')
        axes[2].set_title('Latent Space: Digit Class')
        plt.colorbar(sc3, ax=axes[2], label='Digit (0-9)')
    
    plt.tight_layout()
    plt.show()

# Main execution
n_samples = 10000
input_dim = 28*28
latent_dim = 2
hidden_dim = 256

# Load and transform MNIST data
print("Loading and transforming MNIST data...")
x, rotation_labels, thickness_labels, digit_labels = load_and_transform_mnist(
    n_samples=n_samples, rotations=True, thicknesses=True
)

# Create and train VAE model
model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
print(f"Training VAE with {latent_dim} latent dimensions...")
losses = train_vae(model, x, batch_size=128, n_epochs=50, lr=1e-3)

# Plot loss curves
plt.figure(figsize=(10, 5))
epochs = range(1, len(losses) + 1)
plt.plot(epochs, [l[0] for l in losses], 'b-', label='Total Loss')
plt.plot(epochs, [l[1] for l in losses], 'g-', label='Reconstruction Loss')
plt.plot(epochs, [l[2] for l in losses], 'r-', label='KL Divergence')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot reconstructions
print("Plotting sample reconstructions...")
plot_reconstructions(model, x, n_samples=8)

# Encode all data to get latent representations
model.eval()
with torch.no_grad():
    _, mu, _ = model(x)
z = mu.cpu().numpy()

# Plot latent space
print("Plotting latent space encodings...")
plot_latent_space(z, rotation_labels, thickness_labels, digit_labels)

# %%

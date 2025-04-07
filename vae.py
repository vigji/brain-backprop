# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_signals(n_samples=1000, length=64, frequency_range=(1, 10)):
    signals = []
    amplitude_values = []
    frequency_values = []
    
    for _ in range(n_samples):
        # Random continuous amplitude and frequency
        amp = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(frequency_range[0], frequency_range[1])
        phase = 0  # fixed for orthogonality
        
        # Generate signal
        t = np.linspace(0, 1, length)
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add noise
        signal += np.random.normal(0, 0.2, length)
        
        signals.append(signal)
        amplitude_values.append(amp)
        frequency_values.append(freq)
    
    X = np.vstack(signals)
    return torch.tensor(X, dtype=torch.float32), np.array(amplitude_values), np.array(frequency_values)

class VAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=2, hidden_dim=30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
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
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    # KL divergence between posterior and N(0,1)
    # gamma = 0.05
    gamma = 0.0001
    kl_div = -gamma * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_weight * kl_div, recon_loss, kl_div


def train_vae(model, x, n_epochs=2000, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # KL weight will grow from 0.1 to 1.0 over first half of training
    def get_kl_weight(epoch, start=0.01, end=1.0, steps=n_epochs // 2):
        if epoch < steps:
            return start + (end - start) * (epoch / steps)
        return end

    for epoch in range(n_epochs):
        x_hat, mu, logvar = model(x)
        kl_weight = get_kl_weight(epoch)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch} - Total: {loss.item():.4f} | Recon: {recon.item():.4f} | KL: {kl.item():.4f} | KL weight: {kl_weight:.3f}")

X, amplitude_values, frequency_values = generate_signals()

input_dim = 64
n_samples=1000
frequency_range=(1, 10)
bottleneck_dim = 2
hidden_dim = 30
x, amplitude_labels, smoothness_labels = generate_signals(length=input_dim, 
                                                          frequency_range=frequency_range,
                                                          n_samples=n_samples)
model = VAE(input_dim=input_dim, latent_dim=bottleneck_dim)
# train_model(model, x)
train_vae(model, x, n_epochs=20000, lr=1e-3)

model.eval()
with torch.no_grad():
    _, mu, _ = model(x)
z = mu.numpy()

# %%
# Create figure with three subplots
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 6))


# Plot frequency encoding
sc2 = ax2.scatter(z[:, 0], z[:, 1], c=amplitude_labels, cmap='viridis', alpha=0.7)
ax2.set_xlabel('Latent Dim 1')
ax2.set_ylabel('Latent Dim 2')
ax2.set_title('Latent space: amplitude')
plt.colorbar(sc2, ax=ax2, label='Amplitude (0.5-2.0)')

# Plot phase encoding
sc3 = ax3.scatter(z[:, 0], z[:, 1], c=smoothness_labels, cmap='cividis', alpha=0.7)
ax3.set_xlabel('Latent Dim 1')
ax3.set_ylabel('Latent Dim 2')
ax3.set_title('Latent space: smoothness')
plt.colorbar(sc3, ax=ax3, label='Smooth (0) vs Jagged (1)')

plt.tight_layout()
plt.show()
# %%

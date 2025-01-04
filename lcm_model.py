
import torch
import torch.nn as nn
import torch.nn.functional as F

# Base-LCM Architecture Components
class PreNet(nn.Module):
    """
    Maps input embeddings to the model's hidden dimension after normalization.
    """
    def __init__(self, input_dim, hidden_dim):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.scaler_mean = 0.0  # Placeholder for robust scaler mean
        self.scaler_std = 1.0   # Placeholder for robust scaler std

    def normalize(self, x):
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x):
        x = self.normalize(x)
        x = self.linear(x)
        return x

class PostNet(nn.Module):
    """
    Maps hidden state outputs back to the embedding space with denormalization.
    """
    def __init__(self, hidden_dim, output_dim):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.scaler_mean = 0.0  # Placeholder for robust scaler mean
        self.scaler_std = 1.0   # Placeholder for robust scaler std

    def denormalize(self, x):
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x):
        x = self.linear(x)
        x = self.denormalize(x)
        return x

class TransformerDecoder(nn.Module):
    """
    Standard Decoder-Only Transformer.
    """
    def __init__(self, hidden_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, hidden_dim))  # Positional encoding

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len]
        for layer in self.layers:
            x = layer(x, x)  # Self-attention in decoder layers
        return x

class BaseLCM(nn.Module):
    """
    Base Large Concept Model (LCM):
    - PreNet: Maps input embeddings to hidden space.
    - TransformerDecoder: Autoregressively processes embeddings.
    - PostNet: Maps output back to the embedding space.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim):
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.transformer_decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim)
        self.postnet = PostNet(hidden_dim, output_dim)

    def forward(self, x):
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)
        return x

# Testing the Base-LCM architecture
def test_base_lcm():
    batch_size = 4
    sequence_length = 10
    input_dim = 256  # SONAR embedding dimension (e.g., pre-encoded sentences)
    hidden_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    output_dim = 256  # Output embedding dimension (same as input)

    # Random input to simulate SONAR embeddings
    input_embeddings = torch.randn(batch_size, sequence_length, input_dim)

    # Initialize and test Base-LCM
    model = BaseLCM(input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim)
    output_embeddings = model(input_embeddings)

    print("Input shape:", input_embeddings.shape)
    print("Output shape:", output_embeddings.shape)

if __name__ == "__main__":
    test_base_lcm()

!pip install geoopt

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import PoincareBall, ManifoldParameter  # For hyperbolic embeddings
from geoopt.optim import RiemannianAdam  # Hyperbolic optimizer

# Base-LCM Architecture Components with Hyperbolic Space
class PreNet(nn.Module):
    """
    Maps input embeddings to the model's hidden dimension in hyperbolic space.
    """
    def __init__(self, input_dim, hidden_dim, manifold):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.manifold = manifold
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.linear(x)
        x = self.manifold.expmap0(x)  # Map to hyperbolic space (Poincare Ball)
        return x

class PostNet(nn.Module):
    """
    Maps hidden state outputs back to the embedding space from hyperbolic space.
    """
    def __init__(self, hidden_dim, output_dim, manifold):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.logmap0(x)  # Map back to Euclidean space
        x = self.linear(x)
        return x

class TransformerDecoder(nn.Module):
    """
    Standard Decoder-Only Transformer operating in hyperbolic space.
    """
    def __init__(self, hidden_dim, num_heads, num_layers, ff_dim, manifold, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.manifold = manifold
        self.pos_encoder = ManifoldParameter(torch.zeros(1, 512, hidden_dim), manifold=manifold)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.manifold.expmap0(x + self.pos_encoder[:,:seq_len])  # Ensure curvature is retained
        for layer in self.layers:
            x = layer(x, x)  # Self-attention in decoder layers
        return x

class HyperbolicLCM(nn.Module):
    """
    Base Large Concept Model (LCM) with Hyperbolic Hidden Space.
    - PreNet: Maps input embeddings to hyperbolic space.
    - TransformerDecoder: Operates in hyperbolic space.
    - PostNet: Maps back to Euclidean space.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim, manifold):
        super(HyperbolicLCM, self).__init__()
        self.manifold = manifold
        self.prenet = PreNet(input_dim, hidden_dim, manifold)
        self.transformer_decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim, manifold)
        self.postnet = PostNet(hidden_dim, output_dim, manifold)

    def forward(self, x):
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)
        return x

# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.5):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

# Adding noise to target embeddings
def add_noise_to_embeddings(embeddings, noise_level=0.1):
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise

# Testing the Hyperbolic-LCM Architecture
def test_hyperbolic_lcm():
    batch_size = 4
    sequence_length = 10
    input_dim = 256  # Input embedding dimension
    hidden_dim = 512  # Hidden dimension in hyperbolic space
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    output_dim = 256  # Output embedding dimension
    epochs = 5  # Number of epochs for training
    noise_level = 0.05  # Noise level for targets

    # Initialize the Poincare Ball Manifold
    manifold = PoincareBall(c=1.0)  # Curvature = 1.0

    # Random input to simulate embeddings
    input_embeddings = torch.randn(batch_size, sequence_length, input_dim)

    # Initialize the Hyperbolic-LCM Model
    model = HyperbolicLCM(input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim, manifold)

    # Define the Riemannian Adam optimizer
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Create Target Embeddings with Noise
    target_embeddings = add_noise_to_embeddings(input_embeddings, noise_level=noise_level)

    # Training Loop for Multiple Epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_embeddings = model(input_embeddings)
        loss = criterion(output_embeddings, target_embeddings)
        loss.backward()
        optimizer.step()

        # Compute Accuracy
        accuracy = compute_accuracy(output_embeddings, target_embeddings, threshold=0.2)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_hyperbolic_lcm()

!pip install torchtext

!pip uninstall torchtext --yes
!pip install torchtext --no-cache-dir

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import PoincareBall, ManifoldParameter  # For hyperbolic embeddings
from geoopt.optim import RiemannianAdam  # Hyperbolic optimizer

# Base-LCM Architecture Components with Hyperbolic Space and Pyramid Structure
class PyramidLayer(nn.Module):
    """
    Represents one pyramid layer: compresses dimensionality in hyperbolic space.
    """
    def __init__(self, input_dim, output_dim, manifold):
        super(PyramidLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.expmap0(self.linear(x))  # Map to hyperbolic space with compression
        return x

class HyperbolicCube(nn.Module):
    """
    Hyperbolic Cube: Multiple pyramid layers forming a cube-like structure.
    """
    def __init__(self, layers_dims, manifold):
        super(HyperbolicCube, self).__init__()
        self.manifold = manifold
        self.pyramid_layers = nn.ModuleList([
            PyramidLayer(layers_dims[i], layers_dims[i+1], manifold)
            for i in range(len(layers_dims) - 1)
        ])

    def forward(self, x):
        for layer in self.pyramid_layers:
            x = layer(x)
        return x

class PreNet(nn.Module):
    """
    Maps input embeddings to the hidden dimension.
    """
    def __init__(self, input_dim, hidden_dim, manifold):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.linear(x)
        x = self.manifold.expmap0(x)
        return x

class PostNet(nn.Module):
    """
    Maps output back to the embedding space.
    """
    def __init__(self, hidden_dim, output_dim, manifold):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.linear(x)
        return x

class HyperbolicLCM(nn.Module):
    """
    LCM with a Hyperbolic Cube as the hidden space.
    """
    def __init__(self, input_dim, hidden_dims, num_heads, num_layers, ff_dim, output_dim, manifold):
        super(HyperbolicLCM, self).__init__()
        self.manifold = manifold
        self.prenet = PreNet(input_dim, hidden_dims[0], manifold)
        self.hyperbolic_cube = HyperbolicCube(hidden_dims, manifold)
        self.postnet = PostNet(hidden_dims[-1], output_dim, manifold)

    def forward(self, x):
        x = self.prenet(x)
        x = self.hyperbolic_cube(x)
        x = self.postnet(x)
        return x

# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.1):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

# Load GloVe Embeddings Manually
def load_glove_embeddings(file_path, vocab_size=5000):
    """Load GloVe embeddings from a file for a small subset."""
    embeddings = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings

# Prepare Input Embeddings
def prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=300):
    """Randomly sample embeddings from the loaded GloVe vectors."""
    selected_vectors = torch.stack(
        [glove_embeddings[word] for word in list(glove_embeddings.keys())[:sequence_length]]
    )
    input_embeddings = selected_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
    return input_embeddings

# Testing Hyperbolic-LCM Architecture
def test_hyperbolic_lcm():
    batch_size = 4
    sequence_length = 10
    input_dim = 300  # GloVe embedding dimension
    hidden_dims = [512, 256, 128, 64]  # Pyramid structure dimensions
    output_dim = 300
    epochs = 25
    threshold = 0.1  # Cosine similarity threshold (lowered)
    glove_file = "glove.6B.300d.txt"  # Path to GloVe embeddings file

    # Initialize the Poincare Ball Manifold
    manifold = PoincareBall(c=1.0)

    # Load GloVe Embeddings
    glove_embeddings = load_glove_embeddings(glove_file, vocab_size=100)
    input_embeddings = prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=input_dim)

    # Initialize Hyperbolic-LCM Model
    model = HyperbolicLCM(input_dim, hidden_dims, num_heads=8, num_layers=6, ff_dim=2048, output_dim=output_dim, manifold=manifold)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-4)  # Lowered learning rate
    criterion = nn.MSELoss()

    # Create slightly perturbed target embeddings
    target_embeddings = input_embeddings + torch.randn_like(input_embeddings) * 0.01  # Reduced noise level

    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_embeddings = model(input_embeddings)
        loss = criterion(output_embeddings, target_embeddings)
        loss.backward()
        optimizer.step()

        # Compute Accuracy
        accuracy = compute_accuracy(output_embeddings, target_embeddings, threshold)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_hyperbolic_lcm()

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import PoincareBall, ManifoldParameter  # For hyperbolic embeddings
from geoopt.optim import RiemannianAdam  # Hyperbolic optimizer

# Base-LCM Architecture Components with Hyperbolic Space and Pyramid Structure
class PyramidLayer(nn.Module):
    """
    Represents one pyramid layer: compresses dimensionality in hyperbolic space.
    """
    def __init__(self, input_dim, output_dim, manifold):
        super(PyramidLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.expmap0(self.linear(x))  # Map to hyperbolic space with compression
        return x

class HyperbolicCube(nn.Module):
    """
    Hyperbolic Cube: Multiple pyramid layers forming a cube-like structure.
    """
    def __init__(self, layers_dims, manifold):
        super(HyperbolicCube, self).__init__()
        self.manifold = manifold
        self.pyramid_layers = nn.ModuleList([
            PyramidLayer(layers_dims[i], layers_dims[i+1], manifold)
            for i in range(len(layers_dims) - 1)
        ])

    def forward(self, x):
        for layer in self.pyramid_layers:
            x = layer(x)
        return x

class PreNet(nn.Module):
    """
    Maps input embeddings to the hidden dimension.
    """
    def __init__(self, input_dim, hidden_dim, manifold):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.linear(x)
        x = self.manifold.expmap0(x)
        return x

class PostNet(nn.Module):
    """
    Maps output back to the embedding space.
    """
    def __init__(self, hidden_dim, output_dim, manifold):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.linear(x)
        return x

class HyperbolicLCM(nn.Module):
    """
    LCM with a Hyperbolic Cube as the hidden space.
    """
    def __init__(self, input_dim, hidden_dims, num_heads, num_layers, ff_dim, output_dim):
        super(HyperbolicLCM, self).__init__()
        self.curvature = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # Learnable curvature
        self.manifold = PoincareBall(c=self.curvature)
        self.prenet = PreNet(input_dim, hidden_dims[0], self.manifold)
        self.hyperbolic_cube = HyperbolicCube(hidden_dims, self.manifold)
        self.postnet = PostNet(hidden_dims[-1], output_dim, self.manifold)

    def forward(self, x):
        x = self.prenet(x)
        x = self.hyperbolic_cube(x)
        x = self.postnet(x)
        return x

# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.1):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

# Load GloVe Embeddings Manually
def load_glove_embeddings(file_path, vocab_size=5000):
    """Load GloVe embeddings from a file for a small subset."""
    embeddings = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings

# Prepare Input Embeddings
def prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=300):
    """Randomly sample embeddings from the loaded GloVe vectors."""
    selected_vectors = torch.stack(
        [glove_embeddings[word] for word in list(glove_embeddings.keys())[:sequence_length]]
    )
    input_embeddings = selected_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
    return input_embeddings

# Testing Hyperbolic-LCM Architecture
def test_hyperbolic_lcm():
    batch_size = 4
    sequence_length = 10
    input_dim = 300  # GloVe embedding dimension
    hidden_dims = [512, 256, 128, 64]  # Pyramid structure dimensions
    output_dim = 300
    epochs = 40
    threshold = 0.1  # Cosine similarity threshold (lowered)
    glove_file = "glove.6B.300d.txt"  # Path to GloVe embeddings file

    # Load GloVe Embeddings
    glove_embeddings = load_glove_embeddings(glove_file, vocab_size=100)
    input_embeddings = prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=input_dim)

    # Initialize Hyperbolic-LCM Model
    model = HyperbolicLCM(input_dim, hidden_dims, num_heads=8, num_layers=6, ff_dim=2048, output_dim=output_dim)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-4)  # Lowered learning rate
    criterion = nn.MSELoss()

    # Create slightly perturbed target embeddings
    target_embeddings = input_embeddings + torch.randn_like(input_embeddings) * 0.01  # Reduced noise level

    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_embeddings = model(input_embeddings)
        loss = criterion(output_embeddings, target_embeddings)
        curvature_reg = torch.abs(model.curvature - 1.0) * 0.01  # Regularization term for curvature
        total_loss = loss + curvature_reg
        total_loss.backward()
        optimizer.step()

        # Compute Accuracy
        accuracy = compute_accuracy(output_embeddings, target_embeddings, threshold)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Curvature: {model.curvature.item():.4f} | Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_hyperbolic_lcm()

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import PoincareBall, ManifoldParameter  # For hyperbolic embeddings
from geoopt.optim import RiemannianAdam  # Hyperbolic optimizer

# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.1):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

# Load GloVe Embeddings Manually
def load_glove_embeddings(file_path, vocab_size=5000):
    """Load GloVe embeddings from a file for a small subset."""
    embeddings = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings

# Prepare Input Embeddings
def prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=300):
    """Randomly sample embeddings from the loaded GloVe vectors."""
    selected_vectors = torch.stack(
        [glove_embeddings[word] for word in list(glove_embeddings.keys())[:sequence_length]]
    )
    input_embeddings = selected_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
    return input_embeddings

# Pyramid Layer
class PyramidLayer(nn.Module):
    def __init__(self, input_dim, output_dim, manifold):
        super(PyramidLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.expmap0(self.linear(x))  # Hyperbolic compression
        return x

# Hyperbolic Cube
class HyperbolicCube(nn.Module):
    def __init__(self, layers_dims, manifold):
        super(HyperbolicCube, self).__init__()
        self.manifold = manifold
        self.pyramid_layers = nn.ModuleList([
            PyramidLayer(layers_dims[i], layers_dims[i+1], manifold)
            for i in range(len(layers_dims) - 1)
        ])

    def forward(self, x):
        for layer in self.pyramid_layers:
            x = layer(x)
        return x

# PreNet and PostNet
class PreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, manifold):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.linear(x)
        x = self.manifold.expmap0(x)
        return x

class PostNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, manifold):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.linear(x)
        return x

# Dual Hidden LCM with two hidden dimensions
class DualHiddenLCM(nn.Module):
    def __init__(self, input_dim, hidden_dims, hidden_dim2, output_dim):
        super(DualHiddenLCM, self).__init__()
        self.curvature = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.manifold = PoincareBall(c=self.curvature)

        # Hidden Dimension 1: Pyramid structure
        self.prenet = PreNet(input_dim, hidden_dims[0], self.manifold)
        self.hyperbolic_cube = HyperbolicCube(hidden_dims, self.manifold)
        self.postnet = PostNet(hidden_dims[-1], output_dim, self.manifold)

        # Hidden Dimension 2: 20D bottleneck
        self.hidden_dim2 = nn.Linear(input_dim, hidden_dim2)
        self.hidden_dim2_output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Hidden Dimension 1
        x_hidden1 = self.prenet(x)
        x_hidden1 = self.hyperbolic_cube(x_hidden1)
        x_hidden1 = self.postnet(x_hidden1)

        # Hidden Dimension 2
        x_hidden2 = F.relu(self.hidden_dim2(x))
        x_hidden2 = self.hidden_dim2_output(x_hidden2)

        # Combine outputs
        combined = x_hidden1 + x_hidden2
        return combined

# Testing DualHiddenLCM Architecture
def test_dualhidden_lcm():
    batch_size = 4
    sequence_length = 10
    input_dim = 300  # GloVe embedding dimension
    hidden_dims = [512, 256, 128, 64]  # Pyramid structure dimensions
    hidden_dim2 = 20  # 20D bottleneck
    output_dim = 300
    epochs = 60
    threshold = 0.1  # Cosine similarity threshold
    glove_file = "glove.6B.300d.txt"  # Path to GloVe embeddings file

    # Load GloVe Embeddings
    glove_embeddings = load_glove_embeddings(glove_file, vocab_size=100)
    input_embeddings = prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=input_dim)

    # Initialize DualHiddenLCM Model
    model = DualHiddenLCM(input_dim, hidden_dims, hidden_dim2, output_dim)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Create slightly perturbed target embeddings
    target_embeddings = input_embeddings + torch.randn_like(input_embeddings) * 0.01

    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_embeddings = model(input_embeddings)
        loss = criterion(output_embeddings, target_embeddings)
        curvature_reg = torch.abs(model.curvature - 1.0) * 0.01  # Regularization for curvature
        total_loss = loss + curvature_reg
        total_loss.backward()
        optimizer.step()

        # Compute Accuracy
        accuracy = compute_accuracy(output_embeddings, target_embeddings, threshold)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Curvature: {model.curvature.item():.4f} | Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_dualhidden_lcm()

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from geoopt import PoincareBall, ManifoldParameter
from geoopt.optim import RiemannianAdam

# Initialize DDP
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup DDP
def cleanup_ddp():
    dist.destroy_process_group()

# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.1):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

# DualHiddenLCM Definition
class DualHiddenLCM(nn.Module):
    def __init__(self, input_dim, hidden_dims, hidden_dim2, output_dim):
        super(DualHiddenLCM, self).__init__()
        self.curvature = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.manifold = PoincareBall(c=self.curvature)

        # Hidden Dimension 1: Pyramid structure
        self.prenet = PreNet(input_dim, hidden_dims[0], self.manifold)
        self.hyperbolic_cube = HyperbolicCube(hidden_dims, self.manifold)
        self.postnet = PostNet(hidden_dims[-1], output_dim, self.manifold)

        # Hidden Dimension 2: 20D bottleneck
        self.hidden_dim2 = nn.Linear(input_dim, hidden_dim2)
        self.hidden_dim2_output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Hidden Dimension 1
        x_hidden1 = self.prenet(x)
        x_hidden1 = self.hyperbolic_cube(x_hidden1)
        x_hidden1 = self.postnet(x_hidden1)

        # Hidden Dimension 2
        x_hidden2 = F.relu(self.hidden_dim2(x))
        x_hidden2 = self.hidden_dim2_output(x_hidden2)

        # Combine outputs
        combined = x_hidden1 + x_hidden2
        return combined

# Helper Classes
class PreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, manifold):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.linear(x)
        x = self.manifold.expmap0(x)
        return x

class PostNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, manifold):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.linear(x)
        return x

class HyperbolicCube(nn.Module):
    def __init__(self, layers_dims, manifold):
        super(HyperbolicCube, self).__init__()
        self.manifold = manifold
        self.pyramid_layers = nn.ModuleList([
            PyramidLayer(layers_dims[i], layers_dims[i+1], manifold)
            for i in range(len(layers_dims) - 1)
        ])

    def forward(self, x):
        for layer in self.pyramid_layers:
            x = layer(x)
        return x

class PyramidLayer(nn.Module):
    def __init__(self, input_dim, output_dim, manifold):
        super(PyramidLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.manifold = manifold

    def forward(self, x):
        x = self.manifold.expmap0(self.linear(x))
        return x

# Training Loop
def train(rank, world_size, data, target, input_dim, hidden_dims, hidden_dim2, output_dim, epochs, threshold):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Prepare Model
    model = DualHiddenLCM(input_dim, hidden_dims, hidden_dim2, output_dim).to(device)
    model = DDP(model, device_ids=[rank])

    # Optimizer and Loss
    optimizer = RiemannianAdam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # DataLoader
    dataset = TensorDataset(data, target)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Training
    for epoch in range(epochs):
        model.train()
        for batch_data, batch_target in dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            curvature_reg = torch.abs(model.module.curvature - 1.0) * 0.01  # Regularization
            total_loss = loss + curvature_reg
            total_loss.backward()
            optimizer.step()

        if rank == 0:  # Print only on the main process
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    cleanup_ddp()

# Main Script
def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("This script requires at least 2 GPUs.")
        return

    # Simulated Data
    input_dim = 300
    hidden_dims = [512, 256, 128, 64]
    hidden_dim2 = 20
    output_dim = 300
    epochs = 10
    threshold = 0.1
    batch_size = 4
    num_samples = 100

    data = torch.randn(num_samples, input_dim)
    target = data + torch.randn_like(data) * 0.01  # Slight perturbation

    # Start Training
    torch.multiprocessing.spawn(
        train,
        args=(world_size, data, target, input_dim, hidden_dims, hidden_dim2, output_dim, epochs, threshold),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
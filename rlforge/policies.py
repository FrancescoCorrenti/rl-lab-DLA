import torch
import torch.nn as nn
import torch.nn.functional as F
from rlforge.functional import ACTIVATION_FUNCTIONS
class TemperatureScheduler:
    def __init__(self, initial_temp=1.0, min_temp=0.1):
        self.current_temp = initial_temp
        self.min_temp = min_temp

    def step(self):
        pass

    def get_temperature(self):
        return max(self.current_temp, self.min_temp)

class LinearTemperatureScheduler(TemperatureScheduler):
    def __init__(self, initial_temp=1.0, min_temp=0.1, decay_rate=0.001):
        super().__init__(initial_temp, min_temp)
        self.decay_rate = decay_rate

    def step(self):
        self.current_temp = max(self.current_temp - self.decay_rate, self.min_temp)

class EpsilonScheduler:
    """Simple epsilon decay scheduler for epsilon-greedy policies."""

    def __init__(self, start=1.0, end=0.05, decay=0.995):
        self.epsilon = start
        self.end = end
        self.decay = decay

    def step(self):
        self.epsilon = max(self.end, self.epsilon * self.decay)

    def get_epsilon(self):
        return self.epsilon

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, temperature=2.0):
        super().__init__()
        self.temp_scheduler = LinearTemperatureScheduler(initial_temp=temperature)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits / self.temp_scheduler.get_temperature(), dim=-1)    

    def update_temperature(self):
        self.temp_scheduler.step()

class REINFORCEPolicy(PolicyNetwork):
    def __init__(self, input_dim, output_dim, hidden_dim=128, temperature=1.0):
        super().__init__(input_dim, output_dim, hidden_dim, temperature)

    def forward(self, x):
        # Return raw logits (not probabilities)
        return self.network(x)

class DeepPolicy(PolicyNetwork):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], activation='relu'): # Default hidden_dims updated
        super().__init__(input_dim, output_dim, hidden_dim=hidden_dims[0])
        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(ACTIVATION_FUNCTIONS[activation])
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Proper weight initialization to prevent NaN values
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization to prevent gradient explosion."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for ReLU activations
                nn.init.xavier_uniform_(module.weight)
                # Initialize biases to small positive values
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        # Add input validation to catch NaN inputs early
        if torch.isnan(x).any():
            raise ValueError(f"Input contains NaN values: {x}")
 
        output = self.network(x) / self.temp_scheduler.get_temperature()
        if torch.isnan(output).any():
            raise ValueError(f"Output contains NaN values after forward pass: {output}")
        return output

class ResidualBlock(nn.Module):
    """A residual block that adds input to output every 2 layers."""
    def __init__(self, in_dim, out_dim, activation='relu'):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.layer2 = nn.Linear(out_dim, out_dim)
        self.activation = self._get_activation_fn(activation)
        
        # Projection layer for dimension matching if needed
        self.projection = None
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
    
    def _get_activation_fn(self, activation):
        """Get the activation function based on the string name."""
        if activation in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[activation]
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Supported: {list(ACTIVATION_FUNCTIONS.keys())}")
    
    def forward(self, x):
        identity = x
        
        # Apply first two layers
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        
        # Apply projection if needed for dimension matching
        if self.projection is not None:
            identity = self.projection(identity)
            
        # Add residual connection
        out += identity
        
        # Apply activation after residual connection
        out = self.activation(out)
        
        return out

class DeepResidualPolicy(PolicyNetwork):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 128, 128], activation='relu'): # Default hidden_dims updated
        super().__init__(input_dim, output_dim, hidden_dim=hidden_dims[0])
        
        # Build residual network
        layers = []
        in_dim = input_dim
        
        # Create residual blocks (every 2 layers form a block)
        for i in range(0, len(hidden_dims), 2):
            if i+1 < len(hidden_dims):
                # Two layers with same dimension for clean residual connection
                res_block = ResidualBlock(in_dim, hidden_dims[i], activation)
                layers.append(res_block)
                in_dim = hidden_dims[i]
                
                # Second block with potentially different dimensions
                res_block = ResidualBlock(in_dim, hidden_dims[i+1], activation)
                layers.append(res_block)
                in_dim = hidden_dims[i+1]
            else:
                # Handle odd number of hidden layers
                res_block = ResidualBlock(in_dim, hidden_dims[i], activation)
                layers.append(res_block)
                in_dim = hidden_dims[i]
        
        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        # Replace the network
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with He initialization suitable for residual networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            raise ValueError(f"Input contains NaN values: {x}")        
        output = self.network(x)
        return output


class QNetwork(nn.Module):
    """Simple feed-forward network returning Q-values for each action."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)




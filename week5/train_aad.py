import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import copy
import numpy as np
from torch.utils.data import DataLoader, Subset
from lora_layers import LoRALinear

config = {
    "num_rounds": 10,
    "num_clients": 10,       # Set to 10 for faster local testing on your Mac; syllabus asks for 100 eventually
    "fraction_fit": 1.0,     # Fraction of clients to sample per round
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.01,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "architecture": "FedLoRA_AAD"
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def inject_lora(model, rank, alpha):
    """Convert linear modules to LoRA versions and freeze original weights."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            inject_lora(module, rank, alpha)
    return model

def get_non_iid_dataloaders(num_clients, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    # partition training data by label
    targets = np.array(trainset.targets)
    sorted_indices = np.argsort(targets)
    shard_size = len(trainset) // num_clients
    
    client_loaders = []
    for i in range(num_clients):
        indices = sorted_indices[i * shard_size : (i + 1) * shard_size]
        subset = Subset(trainset, indices)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        
    return client_loaders, testloader

def get_trainable_state_dict(model):
    """Extracts ONLY the parameters that require gradients (LoRA A & B)."""
    trainable_keys = [name for name, param in model.named_parameters() if param.requires_grad]
    state_dict = model.state_dict()
    # Move to CPU for aggregation to save GPU/MPS memory
    return {k: state_dict[k].cpu() for k in trainable_keys}

def client_update(global_model, client_loader, epochs, lr):
    """Perform client-side training, returning LoRA parameters."""
    model = copy.deepcopy(global_model).to(device)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Return ONLY the LoRA weights
    return get_trainable_state_dict(model)

def server_aggregate_aad(global_model, client_weights_list, rank):
    """Aggregation-Aware Decomposition (AAD) via SVD on averaged updates."""
    print("Applying Aggregation-Aware Decomposition (AAD) via SVD...")
    new_state_dict = {}
    
    # Find all unique LoRA layer names (e.g., 'fc1', 'fc2')
    layer_names = set([k.replace('.lora_A', '').replace('.lora_B', '') 
                       for k in client_weights_list[0].keys() if 'lora' in k])
                       
    for layer in layer_names:
        key_A = f"{layer}.lora_A"
        key_B = f"{layer}.lora_B"
        
        avg_delta_w = None
        num_clients = len(client_weights_list)
        
        # reconstruct and average client delta-W
        for client_w in client_weights_list:
            A = client_w[key_A] # Shape: [rank, in_features]
            B = client_w[key_B] # Shape: [out_features, rank]
            
            # \Delta W_i = B_i @ A_i
            delta_w = torch.matmul(B, A) 
            
            if avg_delta_w is None:
                avg_delta_w = delta_w / num_clients
            else:
                avg_delta_w += delta_w / num_clients
                
        # apply SVD
        # U shape: [out_features, out_features], S shape: [min(out, in)], V shape: [in_features, in_features]
        U, S, V = torch.svd(avg_delta_w)
        
        # truncate and rebuild A/B
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:, :rank] # Note: PyTorch SVD returns V, not V^T
        
        # B = U_r * sqrt(S_r)
        # A = sqrt(S_r) * V_r^T
        sqrt_S = torch.diag(torch.sqrt(S_r))
        new_B = torch.matmul(U_r, sqrt_S)
        new_A = torch.matmul(sqrt_S, V_r.t())
        
        new_state_dict[key_B] = new_B
        new_state_dict[key_A] = new_A
        
    # update global model
    global_model.load_state_dict(new_state_dict, strict=False)

def evaluate(model, testloader):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    wandb.init(project="cifar10-fedlora", config=config)
    print(f"Running on device: {device}")

    client_loaders, testloader = get_non_iid_dataloaders(config["num_clients"], config["batch_size"])
    
    # Initialize Global Model
    global_model = SimpleCNN()
    global_model = inject_lora(global_model, rank=config["lora_rank"], alpha=config["lora_alpha"])
    
    # Calculate bytes for WandB reporting
    lora_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    full_params = sum(p.numel() for p in global_model.parameters())
    # 4 bytes per float32 parameter
    print(f"Communication cost per client per round (FedLoRA): {(lora_params * 4) / 1024 / 1024:.2f} MB")
    print(f"Communication cost per client per round (Standard FedAvg): {(full_params * 4) / 1024 / 1024:.2f} MB")

    for round_num in range(config["num_rounds"]):
        print(f"\n--- Global Round {round_num + 1}/{config['num_rounds']} ---")
        
        client_weights = []
        num_sampled_clients = max(1, int(config["fraction_fit"] * config["num_clients"]))
        
        for client_idx in range(num_sampled_clients):
            print(f"Training client {client_idx + 1}/{num_sampled_clients}...")
            # Clients pull the global model, train on local data, and return ONLY LoRA weights
            local_weights = client_update(global_model, client_loaders[client_idx], 
                                          config["local_epochs"], config["learning_rate"])
            client_weights.append(local_weights)
            
        # Server aggregates LoRA weights and updates global model
        print("Aggregating LoRA weights on server...")
        global_model = server_aggregate_aad(global_model, client_weights, rank=config["lora_rank"])
        
        # Evaluate global model
        val_acc = evaluate(global_model, testloader)
        print(f"Round {round_num + 1} Global Test Accuracy: {val_acc:.2f}%")
        
        wandb.log({
            "round": round_num + 1, 
            "global_test_accuracy": val_acc,
            "cumulative_communication_MB": (round_num + 1) * num_sampled_clients * ((lora_params * 4) / 1024 / 1024)
        })

    print("Federated Training Complete.")
    wandb.finish()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from lora_layers import LoRALinear

config = {
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.01,
    "architecture": "SimpleCNN_LoRA",
    "dataset": "CIFAR-10",
    "lora_rank": 8,
    "lora_alpha": 16.0
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

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

# LoRA Injection and Parameter Counting Functions
def inject_lora(model, rank, alpha):
    """Replaces linear layers with LoRA layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            inject_lora(module, rank, alpha)
    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    efficiency = 100 * trainable_params / all_param
    print(f"Trainable Params: {trainable_params} | Total Params: {all_param} | Trainable %: {efficiency:.4f}%")
    return trainable_params, all_param, efficiency

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()



model = inject_lora(model, rank=config["lora_rank"], alpha=config["lora_alpha"])
model = model.to(device)


train_p, all_p, eff_percent = print_trainable_parameters(model)
config["trainable_parameters"] = train_p
config["total_parameters"] = all_p
config["parameter_efficiency_percent"] = eff_percent

wandb.init(project="cifar10-week3", config=config)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])

print(f"Training on {device}...")
model.train()

for epoch in range(config['epochs']):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 99:
            wandb.log({
                "epoch": epoch + 1,
                "loss": running_loss / 100,
                "gradient_norm": total_norm
            })
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0


    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    wandb.log({"val_accuracy": val_acc})
    print(f'Accuracy of the network on test images: {val_acc:.2f} %')
    model.train() # Set back to train mode for next epoch

print('Finished Training')
wandb.finish()
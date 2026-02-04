import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import wandb

# --- CONFIGURATION ---
args = {
    "num_clients": 100,      # Total clients in the system
    "fraction": 0.1,         # C: Fraction of clients to select per round (10 clients)
    "local_ep": 5,           # E: Number of local epochs per client
    "local_bs": 10,          # B: Local batch size
    "lr": 0.01,              # Learning rate (FedAvg usually needs higher LR than centralized)
    "rounds": 50,            # Global communication rounds
    "dataset": "cifar10",
    "partition": "noniid",
        'gpu': -1   
}


wandb.init(project="cifar10-week2", config=args, name="fedavg_noniid")

def get_noniid_dataset(dataset, num_users):
    """
    Sorts data by label, splits into shards, and assigns 2 shards to each client.
    This creates a highly Non-IID distribution.
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

  
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    
    num_shards = 200
    num_imgs = len(dataset) // num_shards # 250 images per shard
    idx_shard = [i for i in range(num_shards)]
    
    for i in range(num_users):
      
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users

from torch.utils.data import Dataset, DataLoader

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

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
    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                    batch_size=self.args['local_bs'], shuffle=True)

    def train(self, net):
        net.train()
      
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['lr'], momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        
        epoch_loss = []
        for iter in range(self.args['local_ep']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(device), labels.to(device) 
                net.zero_grad()
                log_probs = net(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['local_bs'])
    l = len(data_loader)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args['gpu'] != -1:
                data, target = data.to(device), target.to(device)
            log_probs = net_g(data)
     
            test_loss += nn.CrossEntropyLoss()(log_probs, target).item()
            
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=trans)
dataset_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=trans)


dict_users = get_noniid_dataset(dataset_train, args['num_clients'])


net_glob = SimpleCNN().to(device) 
net_glob.train()


w_glob = net_glob.state_dict()


for iter in range(args['rounds']):
    w_locals, loss_locals = [], []
    
   
    m = max(int(args['fraction'] * args['num_clients']), 1)
    idxs_users = np.random.choice(range(args['num_clients']), m, replace=False)
    
    print(f"Round {iter+1}/{args['rounds']} - Clients: {idxs_users}")


    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
    

    w_glob = FedAvg(w_locals)
    
   
    net_glob.load_state_dict(w_glob)


    loss_avg = sum(loss_locals) / len(loss_locals)
    

    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args) # (Define a simple test func)
    
    print(f"Round {iter} - Average Loss: {loss_avg:.3f}, Test Acc: {acc_test:.2f}")
    

    wandb.log({"Round": iter, "Loss": loss_avg, "Test Accuracy": acc_test})
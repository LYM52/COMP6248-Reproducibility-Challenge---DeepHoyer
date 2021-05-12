import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from net.models import LeNet_5 as LeNet
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')                
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12345678, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--model', type=str, default='saves/initial_model',
                    help='path to model pretrained with sparsity-inducing regularizer')                    
parser.add_argument('--sensitivity', type=float, default=0.25,
                    help="pruning threshold computed as sensitivity value multiplies to layer's std")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = LeNet(mask=False).to(device)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    best = 0.0
    pbar = tqdm(range(epochs), total=epochs)
    for epoch in pbar:
        accuracy = test()
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), args.model+'_V_'+str(args.sensitivity)+'.pth')
            
        model.train()    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss = loss
                
            total_loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Best: {best:.2f}% Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f} Total: {total_loss.item():.6f}')
    print(f'Accuracy: {best:.2f}%')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        #print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


model.load_state_dict(torch.load(args.model+'.pth'))
# Initial training
print("--- Pruning ---")
for name, p in model.named_parameters():
    if 'mask' in name:
        continue
    tensor = p.data.cpu().numpy()
    threshold = args.sensitivity*np.std(tensor)
    print(f'Pruning with threshold : {threshold} for layer {name}')
    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
    p.data = torch.from_numpy(new_mask).to(device)

accuracy = test()
fc1, fc2 = util.print_nonzeros(model)

print("--- Finetuning ---")
train(args.epochs)
print("想要的结果")
fc1,fc2 = util.print_nonzeros(model)
print("fc1_new",len(fc1))
print("fc1_new_none",np.count_nonzero(fc1))
print("fc2_new",len(fc2))
print("fc2_new_none",np.count_nonzero(fc2))
fc1_new = np.round(fc1,4)
fc2_new = np.round(fc2,4)
#ef fcPrint():
#   return fc1,fc2
fc1_new = fc1_new.tolist()
fc2_new = fc2_new.tolist()
print("fc1",fc1_new)
print("fc2",fc2_new)

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#ax[0].hist(fc1_new, bins=20)
#ax[0].hist(fc2_new, bins=20)
#plt.savefig("c9.png")

fc1_new = np.round(fc1,4)
fc2_new = np.round(fc2,4)
fc1_new = fc1_new.tolist()
fc2_new = fc2_new.tolist()
result1 = dict()
for a  in set(fc1_new):
    result1[a] = fc1_new.count(a)
print("result1",result1)


key_value1 = list(result1.keys())
key_value1 = np.array(key_value1)
key_value1 = np.round(key_value1,4)
key_value1 = key_value1.tolist()
value_list1 = list(result1.values())
print("key_value1",key_value1)
result2 = dict()
for a  in set(fc2_new):
    result2[a] = fc2_new.count(a)

key_value2 = list(result2.keys())
key_value2 = np.array(key_value2)
key_value2 = np.round(key_value2,4)
key_value2 = key_value2.tolist()

value_list2 = list(result2.values())
print("key_value1",key_value2)
print("value_list2",value_list2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4)) 
ax[0].scatter(key_value1, value_list1, c="m", s=3) 
ax[1].scatter(key_value2, value_list2, c="m", s=3) 


plt.savefig("c10.png")




#result1.pop(0)

#key_value3 = list(result1.keys())
#key_value3 = np.array(key_value3)
#key_value3 = np.round(key_value3,4)
#key_value3 = key_value3.tolist()
#value_list3 = list(result1.values())
#print("key_value3",key_value3)

#result2.pop(0)

#key_value4 = list(result2.keys())
#key_value4 = np.array(key_value4)
#key_value4 = np.round(key_value4,4)
#key_value4 = key_value4.tolist()

#value_list4 = list(result2.values())
#print("key_value4",key_value4)
#print("value_list4",value_list4)

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4)) 
#ax[0].scatter(key_value3, value_list3, c="m", s=3) 
#ax[1].scatter(key_value4, value_list4, c="m", s=3) 


#plt.savefig("c11.png")







#def fcPrintp():
#   return key_value1,key_value2

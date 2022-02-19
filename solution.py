import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path
from torchvision import utils

import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# global constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 3
NUM_CLASSES = 10
HIDDEN_SIZE = [128, 512, 512, 512, 512, 512]
NUM_EPOCHS = 20  # default is 20, changeable via cl
BATCH_SIZE = 200
LR = 2e-3
LR_DECAY = 0.95
REG = 0.001
TRAINING_SIZE = 49000
VAL_SIZE = 1000
DROP_OUT = 0.2

# these helper functions are provided for model weights initialization and for updating model's parameters
def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def PrintModelSize(model, disp=True):
    # TODO: Implement the function to count the number of trainable parameters in
    #  the input model.

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def VisualizeFilter(model):
    # TODO: Implement the functiont to visualize the weights in the first conv layer in the model.
    #  Visualize them as a single image of stacked filters.
    #  You can make use of the torchvision.utils.make_grid and
    #  matlplotlib.imshow to visualize an image in python.
    
    kernels = model.conv1.weight.detach().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img = torchvision.utils.make_grid(kernels, nrow = 15)
    plt.imshow(filter_img.permute(1, 2, 0))
    


def get_cifar10_dataset(val_size=VAL_SIZE, batch_size=BATCH_SIZE):
    """
    Load and transform the CIFAR10 dataset. Make Validation set. Create dataloaders for
    train, test, validation sets. Only train_loader uses batch_size of 200, val_loader and
    test_loader have 1 batch (i.e. batch_size == len(val_set) etc.)

    DO NOT CHANGE THE CODE IN THIS FUNCTION. YOU MAY CHANGE THE BATCH_SIZE PARAM IF NEEDED.

    If you get an error related num_workers, you may change that parameter to a different value.

    :param val_size: size of the validation partition
    :param batch_size: number of samples in a batch
    :return:
    """

    # the datasets.CIFAR getitem actually returns img in PIL format
    # no need to get to Tensor since we're working with our own model and not PyTorch
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703233, 0.24348505, 0.26158768))
                                    ])

    # Load the train_set and test_set from PyTorch, transform each sample to a flattened array
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    classes = train_set.classes

    # Split data and define train_loader, test_loader, val_loader
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                              shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_size,
                                             shuffle=False, num_workers=2)

    return train_loader, test_loader, val_loader, classes


# TODO: Implement the class ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 512, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        
        self.bn = nn.BatchNorm2d(512)
        self.dropout2d = nn.Dropout2d(0.2)
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.bn(x)
        x = self.relu(self.pool(self.conv3(x)))
        x = self.dropout2d(self.bn(x))
        x = self.relu(self.pool(self.conv4(x)))
        x = self.bn(x)
        x = self.relu(self.pool(self.conv5(x)))
        x = self.dropout2d(self.bn(x))
        x = x.view(-1, 512)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
    
# TODO: Implement the forward pass computations

# TODO: Implement the training function
# TODO: Implement the evaluation function

def train(trainloader, optimizer, net, criterion, device):
    running_loss = 0.0
    net.train()
    accuracy = 0
    total = 0
    for i, data in enumerate(trainloader):
        inputs = data[0].to(device)
        total+=inputs.shape[0]
        labels = data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        accuracy+=calculate_correct_tag_num(outputs,labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss/len(trainloader),accuracy/total

def validate(dataloader, net, criterion, device):
    running_loss = 0.0
    net.eval()
    accuracy = 0
    total = 0
    for i, data in enumerate(dataloader):
        inputs = data[0].to(device)
        total+=inputs.shape[0]
        labels = data[1].to(device)
        with torch.no_grad():
            outputs = net(inputs)
        accuracy+=calculate_correct_tag_num(outputs,labels)
            
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
    return running_loss/len(dataloader),accuracy/total

def train_validate(epochs = 5):
    train_loader, test_loader, val_loader, classes = get_cifar10_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=0.001)

    for i in range(epochs):
        print("Epoch : ", i+1)
        train_loss,train_accuracy = train(train_loader, optimizer, net, criterion, device)
        valid_loss,valid_accuracy = validate(val_loader, net, criterion, device)
        print("Train loss : ", train_loss)
        print("Train accuracy: ",train_accuracy)
        print("Valid loss : ", valid_loss)
        print("Valid accuracy： ", valid_accuracy)
        
    print("Training Completed")
    return net

def calculate_correct_tag_num(prediction,y):
    prediction = torch.max(prediction,1)[1]
    correct = 0
    for i,j in zip(prediction,y):
        if i==j:
            correct+=1

    return correct

def test(net):
    # Testing on cpu #
    _, test_loader, _, _ = get_cifar10_dataset()
    device = 'cpu'
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_accuracy = validate(test_loader, net, criterion, device)
    print("_____________________________________________________________")
    print("Test loss : ", test_loss)
    print("Test accuracy : ", test_accuracy)
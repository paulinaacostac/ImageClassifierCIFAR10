import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# python image library of range [0,1]
# transform them to tensors of normalized range [-1,1]

transform = transforms.Compose( # composing several transforms together
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] #mean = 0.5 std = 0.5
)

batch_size = 4
num_workers = 2
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy() # convert to numpy objects
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter = iter(trainloader) # get random training images with iter function
images,labels = dataiter.next()

#call function on our images
imshow(torchvision.utils.make_grid(images))

#print the class of the image
print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
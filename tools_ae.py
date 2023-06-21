import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# print memory usage
def memory_usage():
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    print(f'Cached:   {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')

# load dataset
def loaddataset(your_label, b_size=4):
    # load the dataset for training
    transformer = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
    
    
    data_dir = '/home/wxia/researchdrive/exp/external_dataset/horse2zebra/' 
    dataset = datasets.ImageFolder(data_dir, transform=transformer)
    
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    print(f'Cached:   {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')

    filtered_data = [sample for sample in dataset if sample[1] == your_label]
    data_loader = DataLoader(filtered_data, batch_size=b_size, shuffle=True, num_workers=2)
    
    print('finish loading the dataset')
    return data_loader

def loadimage(dataloader, path='./printed_images', num=2):
    # print out some data
    iterator = iter(dataloader)
    images, labels = next(iterator)
    fig, axis = plt.subplots(num, num, figsize=(num, num))
    images = images[:num*num].detach().cpu().numpy()
    for i in range(num*num):
        x = images[i]/2+0.5
        ax = axis[i//num][i%num]
        ax.imshow(x.transpose(1,2,0))
        ax.axis('off')
    # path = './printed_images'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('./printed_images/showcases.png')
    plt.show()

def printimage(images, path='./printed_images', name='showcases', num=2):
    fig, axis = plt.subplots(num, num, figsize=(num, num))
    images = images[:num*num].detach().cpu().numpy()
    for i in range(num*num):
        x = images[i]/2+0.5
        ax = axis[i//num][i%num]
        ax.imshow(x.transpose(1,2,0))
        ax.axis('off')
    # path = './printed_images'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('./printed_images/name_{}.png'.format(name))
    plt.show()
    
    
def training():
    pass

# # load training datas
# dataloader = loaddataset(0)

# # print some images
# loadimage(dataloader, path='./printed_images/train_horse', num=4)
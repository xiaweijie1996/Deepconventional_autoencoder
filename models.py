import torch 
import torch.nn as nn

# define the reshape layer
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape) # upack the tuple


class Encoder(nn.Module):
    def __init__(self, h_picture):
        super(Encoder, self).__init__()
        
        self.h_picture = h_picture
        
        self.seq = nn.Sequential(  
            
            # first conv layer                      
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # third conv layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            
            # reshape layer
            Reshape((-1, 3*self.h_picture*self.h_picture)),
            
            # linear layer
            nn.Linear(3*self.h_picture*self.h_picture, 512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
    def forward(self, x):
        return self.seq(x)
        
class Decoder(nn.Module):
    def __init__(self, h_picture):
        super(Decoder, self).__init__()
        
        self.h_picture = h_picture
        
        self.seq = nn.Sequential(
            
            # linear layer
            nn.Linear(512, 3*self.h_picture*self.h_picture),
            nn.BatchNorm1d(3*self.h_picture*self.h_picture),
            nn.LeakyReLU(),
            
            # reshape layer
            Reshape((-1, 3, self.h_picture, self.h_picture)),
            
            # transpose conv layer 1
            nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # transpose conv layer 2
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # laset transpose conv layer
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.seq(x)
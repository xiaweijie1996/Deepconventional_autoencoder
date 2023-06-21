# %%
import tools_ae as tl
import models as md
import os 
import torch
import matplotlib.pyplot as plt

# select the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test the 
# load model
encoder = md.Encoder(128).to(device)
decoder = md.Decoder(128).to(device)
encoder.load_state_dict(torch.load('./model/encoder.pth'))
decoder.load_state_dict(torch.load('./model/decoder.pth'))

# load test datas
data_dir = '/home/wxia/researchdrive/exp/external_dataset/horse2zebra/'
dataloader = tl.loaddataset(data_dir,3,9) # 3 is the label of zerbra/2 is the label of horse

#%%
for i, (img, num_label) in enumerate(dataloader):
    # forward
    z = encoder(img.to(device))
    x = decoder(z)
    
    # print the image
    tl.printimage(x, name='test_horse_generated', num=3)
    tl.printimage(img, name='test_horse_original', num=3)
    break

# %%

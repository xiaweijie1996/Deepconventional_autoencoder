#%%
import tools_ae as tl
import models as md
import os 
import torch.optim as optim
import torch

print(os.getcwd())

# select the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tl.memory_usage()

# load training datas
dataloader = tl.loaddataset(0) # 1 is the label of zerbra/0 is the label of horse
tl.memory_usage()

# print some images
# tl.loadimage(dataloader, path='./printed_images/train_horse')

#%%
# load model
encoder = md.Encoder(128).to(device)
decoder = md.Decoder(128).to(device)
print(encoder)
print('-------------------')
print(decoder)
tl.memory_usage()

# hyperparameters
lr = 0.0001
epoches = 10000

# optimizer 
optimizer_e = optim.Adam(encoder.parameters(), lr=lr)
optimizer_d = optim.Adam(decoder.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# record loss
re_con_losses = []

#%%
# # training
# encoder.load_state_dict(torch.load('./model/encoder.pth'))
# decoder.load_state_dict(torch.load('./model/decoder.pth'))

for epoch in range(epoches):
    for i, (img, num_label) in enumerate(dataloader):
        # train the decoder
        optimizer_d.zero_grad()
        optimizer_e.zero_grad()
        
        # forward
        z = encoder(img.to(device))
        x = decoder(z)
        
        # loss
        loss = criterion(x, img.to(device))
        
        # backward
        loss.backward()
        optimizer_d.step()
        optimizer_e.step()
        
        # record loss
        re_con_losses.append(loss.item())
        
        # print the loss
    
    print('epoch: %d, loss: %f'%(epoch, loss.item()))
    tl.memory_usage()
    
# save the model
os.makedirs('./model', exist_ok=True)
torch.save(encoder.state_dict(), './model/encoder.pth')
torch.save(decoder.state_dict(), './model/decoder.pth')


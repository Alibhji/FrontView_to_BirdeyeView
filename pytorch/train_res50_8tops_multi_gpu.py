# this code is  written by Ali Babolhaveji
# convert front view to birdeye view
# Pytorch + GPU
# 3/11/2020


# command :
# python -W ignore train_res50_8tops.py 

from lib import Bkend_res50_8top
from lib import Dataset_top_to_birdView
from lib import train_ , validation_

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import shutil
import pickle
import pandas as pd
import gc






from torch.utils.tensorboard import SummaryWriter
SummaryWriter._PRINT_DEPRECATION_WARNINGS = False
import torchvision



history = pd.DataFrame()

start_epoch     = 0
end___epoch     = 100

train_batch     = 8
val_batch       = 8
num_workers     = 40

learning_rate=0.001
experiment_name = 'experiment_1'+f'_epoch_{start_epoch}to_E{end___epoch}_batch_{train_batch}_lr_{learning_rate}'
resualt_save_dir        = os.path.join('runs',experiment_name)
del_dir = input(f"Do you want to delet [{resualt_save_dir}] directory? (y/n)")

if(del_dir=='y'):
    if(os.path.exists(resualt_save_dir)):
        shutil.rmtree(resualt_save_dir)
else:
    assert del_dir=='y' , 'the program is stoped.'

model_save_dir          = os.path.join(resualt_save_dir , 'saved_models')
os.makedirs(model_save_dir)  

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1,2,3"


best_loss = 1000000000;




# Creaete Model
model = Bkend_res50_8top()

# deefine loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# t_in = torch.randn(1,3,224,224 )
# t_in2 = torch.randn(1,4)
# model(t_in,t_in2).shape

# Create Datasets
train_list =  ['./data/train/000' ]

val_list = ['./data/val/000' ,'./data/val/001' ]

training_generator     = Dataset_top_to_birdView( train_list ,type_='train' ,check_images =True)
validation_generator   = Dataset_top_to_birdView( val_list ,type_='val' ,check_images =True)

# create Dataloader


train_loader = DataLoader(training_generator  , batch_size = train_batch ,num_workers = num_workers ,shuffle =True , pin_memory =True)
val_loader   = DataLoader(validation_generator, batch_size = val_batch   ,num_workers = num_workers)


#create tensorbordx Logger
writer = SummaryWriter(resualt_save_dir)

# https://github.com/lanpa/tensorboardX

# get some random training images save model architecture and dataset sample
dataiter = iter(train_loader)
label_front, crop_front ,label_top ,meta_data = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(crop_front)

writer.add_image('training_set_batches', img_grid)
writer.add_graph(model,  (crop_front ,label_front))
writer.close()

# Transfer model on the GPU/GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

torch.cuda.set_device(0)
model = model.to(device) 
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model ,device_ids=[0,1,2,3])

    
  




for epoch in range(start_epoch ,end___epoch):  # loop over the dataset multiple times
    print(f'=======================  Epoch {epoch} / {end___epoch}  =======================')
    gc.collect()
    
    train_(model ,
           train_loader ,
           epoch,
           device=device,criterion=criterion ,
           optimizer=optimizer,
           writer=writer)
    
    curr_val_loss =\
    validation_ (model 
                 ,val_loader , 
                 epoch ,device=device,
                 criterion=criterion ,
                 writer=writer)
    
    for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    writer.close()
#     print('curr_val_loss' , curr_val_loss)
    
    

    if curr_val_loss  < best_loss:
        model_save_format = f"model_E{epoch:03d}_Loss{curr_val_loss:.6f}.pt"
        #torch.save(model.state_dict(), os.path.join(experiment_name ,f"./model_E{epoch:03d}_Loss{curr_val_loss:.6f}.pt"))
        torch.save(model.state_dict(), os.path.join(model_save_dir , model_save_format))
        #print (f"./model_E{epoch:03d}_Loss{curr_val_loss:.6f}.pt  is saved.")
        print (os.path.join(model_save_dir , model_save_format) + " is saved.")
        best_loss = curr_val_loss
        with open(os.path.join(model_save_dir , f'history.pkl'), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    

with open(os.path.join(model_save_dir , f'history_{start_epoch}_{end___epoch}.pkl'), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Finished Training')

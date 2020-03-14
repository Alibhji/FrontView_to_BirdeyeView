# r = tqdm(train_loader)

#from tqdm.notebook import tqdm
from tqdm import tqdm
import copy
import time
from ..utils import IOU ,mAP
import pandas as pd

Meta_Data = pd.DataFrame()

def update_meta_data(meta_data,label_top, label_front, outputs ):
    global Meta_Data
    
    tops = zip( (outputs.cpu().detach().numpy().tolist()),(label_top.cpu().detach().numpy()).tolist())
    Iou_list =[]

    for itm in tops:
        Iou_list.append(IOU(*(itm)))

    temp = pd.DataFrame()
    meta_data.update({'iou':Iou_list})
    meta_data.update({'label_top'    :label_top.cpu().detach().numpy().tolist()})
    meta_data.update({'label_top_pre':outputs.cpu().detach().numpy().tolist()})
    meta_data.update({'label_front' :label_front.cpu().detach().numpy().tolist()})
    Meta_Data = Meta_Data.append(pd.DataFrame(meta_data) ,ignore_index=True)





def train_(model ,train_loader , epoch, device, criterion, optimizer, writer, history =None):
    global Meta_Data 
    Meta_Data  = pd.DataFrame()
    ep_since = time.time()  
    model.train()
    running_loss = 0.0
    r = tqdm(train_loader)
    
    mAp_ = 0

    for i, data in enumerate(r, 0):
        since = time.time()
        # get the inputs; data is a list of [inputs, labels]
        label_front, crop_front ,label_top, meta_data = data
        label_front =label_front.to(device)
        crop_front =crop_front.to(device)
        label_top =label_top.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(crop_front,label_front)
        loss = criterion(outputs, label_top)
        loss.backward()
        optimizer.step()

        update_meta_data(meta_data,label_top, label_front, outputs )

        
        batch_loss = copy.deepcopy(loss.item())
        batch_loss = batch_loss / label_front.shape[0]
        
        writer.add_scalar('training batch loss',
                        batch_loss ,
                        epoch * len(train_loader) + i)
        time_elapsed_batch = time.time() - since
        time_elapsed = time.time() - ep_since

        if history is not None:
            history.loc[epoch * len(train_loader) + i, 'train_Batch_loss'] = batch_loss
            #history.loc[epoch * len(train_loader) + i, 'time'] = time_elapsed
        

        # print statistics
        running_loss += loss.item() / len(train_loader)
        
        
        r.set_description(f'([T/{epoch}](L: {running_loss:0.6f} , BL{batch_loss: 0.6f} ,time {time_elapsed_batch: 03.3f} {time_elapsed: 03.3f})')
        if (i == len(train_loader)-1):
            mAp_50 = mAP(Meta_Data ,.50) * 100
            mAp_75 = mAP(Meta_Data ,.75) * 100
            mAp_90 = mAP(Meta_Data ,.90) * 100
            r.set_description(f'([T/{epoch}](L: {running_loss:0.6f} , BL{batch_loss: 0.6f} ,time {time_elapsed_batch: 03.3f} {time_elapsed: 03.3f} [mAP_50: {mAp_50:02.3f}%, mAP_75: {mAp_75:02.3f}%, mAP_90: {mAp_90:02.3f}%]')
#         writer.close()
        
    writer.add_scalar('training loss',
                        running_loss ,
                        epoch )
    writer.add_scalars('mAP', {'mAP_50%' : mAp_50,
                                'mAP_75%': mAp_75,
                                'mAP_90%': mAp_90}, epoch)
    writer.close()
    if history is not None:
            history.loc[epoch, 'train_loss'] = running_loss
            history.loc[epoch, 'mAp_50'] = mAp_50  
            history.loc[epoch, 'mAp_75'] = mAp_50 
            history.loc[epoch, 'mAp_900'] = mAp_50  
            history.loc[epoch, 'train_time'] =  time.time() - ep_since
    


        
    
    
def validation_ (model ,val_loader ,epoch , device, criterion, writer, history =None):
    global Meta_Data 
    Meta_Data  = pd.DataFrame()
    ep_since = time.time() 
    model.eval()
    running_loss = 0.0
    r2 = tqdm(val_loader)
    for i, data in enumerate(r2, 0):
        since = time.time()
        # get the inputs; data is a list of [inputs, labels]
        label_front, crop_front ,label_top ,meta_data = data
        label_front =label_front.to(device)
        crop_front =crop_front.to(device)
        label_top =label_top.to(device)

        # forward + backward + optimize
        outputs = model(crop_front,label_front)
        loss = criterion(outputs, label_top)
        update_meta_data(meta_data,label_top, label_front, outputs )
        
        batch_loss = copy.deepcopy(loss.item())
        batch_loss = batch_loss / label_front.shape[0]
        
        writer.add_scalar('validation batch loss',
                        batch_loss ,
                        epoch * len(val_loader) + i)
        time_elapsed_batch = time.time() - since
        time_elapsed = time.time() - ep_since

        if history is not None:
            history.loc[epoch * len(train_loader) + i, 'val_Batch_loss'] = batch_loss
        

        # print statistics
        running_loss += loss.item() / len(val_loader)
        
        r2.set_description(f' [E/{epoch}](L: {running_loss:0.6f} , BL{loss.item() / label_front.shape[0]: 0.6f} ,time {time_elapsed_batch: 03.3f} {time_elapsed: 03.3f})')
        if (i == len(val_loader)-1):
            mAp_50 = mAP(Meta_Data ,.50) * 100
            mAp_75 = mAP(Meta_Data ,.75) * 100
            mAp_90 = mAP(Meta_Data ,.90) * 100
            r2.set_description(f'([T/{epoch}](L: {running_loss:0.6f} , BL{batch_loss: 0.6f} ,time {time_elapsed_batch: 03.3f} {time_elapsed: 03.3f} [mAP_50: {mAp_50:02.3f}%, mAP_75: {mAp_75:02.3f}%, mAP_90: {mAp_90:02.3f}%]')
        
    writer.add_scalar('validation loss',
                        running_loss ,
                        epoch )
    writer.add_scalars('mAP', {'mAP_50%' : mAp_50,
                                'mAP_75%': mAp_75,
                                'mAP_90%': mAp_90}, epoch)
    writer.close()
    if history is not None:
            history.loc[epoch, 'val_loss'] = running_loss
            history.loc[epoch, 'mAp_50'] = mAp_50  
            history.loc[epoch, 'mAp_75'] = mAp_50 
            history.loc[epoch, 'mAp_900'] = mAp_50  
            history.loc[epoch, 'val_time'] =  time.time() - ep_since
        
    return running_loss
        

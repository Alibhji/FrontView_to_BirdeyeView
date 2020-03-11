# r = tqdm(train_loader)

#from tqdm.notebook import tqdm
from tqdm import tqdm
import copy


def train_(model ,train_loader , epoch, device, criterion, optimizer, writer, history =None):
    model.train()
    running_loss = 0.0
    r = tqdm(train_loader)
    for i, data in enumerate(r, 0):
        # get the inputs; data is a list of [inputs, labels]
        label_front, crop_front ,label_top = data
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
        
        batch_loss = copy.deepcopy(loss.item())
        batch_loss = batch_loss / label_front.shape[0]
        
        writer.add_scalar('training batch loss',
                        batch_loss ,
                        epoch * len(train_loader) + i)

        if history is not None:
            history.loc[epoch * len(train_loader) + i, 'train_Batch_loss'] = batch_loss
        

        # print statistics
        running_loss += loss.item() / len(train_loader)
        
        r.set_description(f'([T/{epoch}](L: {running_loss:0.6f} , BL{batch_loss: 0.6f})')
        
#         writer.close()
        
    writer.add_scalar('training loss',
                        running_loss ,
                        epoch )
    writer.close()
    if history is not None:
            history.loc[epoch, 'train_loss'] = running_loss
        
    
    
def validation_ (model ,val_loader ,epoch , device, criterion, writer, history =None):
    model.eval()
    running_loss = 0.0
    r2 = tqdm(val_loader)
    for i, data in enumerate(r2, 0):
        # get the inputs; data is a list of [inputs, labels]
        label_front, crop_front ,label_top = data
        label_front =label_front.to(device)
        crop_front =crop_front.to(device)
        label_top =label_top.to(device)

        # forward + backward + optimize
        outputs = model(crop_front,label_front)
        loss = criterion(outputs, label_top)
        
        batch_loss = copy.deepcopy(loss.item())
        batch_loss = batch_loss / label_front.shape[0]
        
        writer.add_scalar('validation batch loss',
                        batch_loss ,
                        epoch * len(val_loader) + i)

        if history is not None:
            history.loc[epoch * len(train_loader) + i, 'val_Batch_loss'] = batch_loss
        

        # print statistics
        running_loss += loss.item() / len(val_loader)
        
        r2.set_description(f' [E/{epoch}](L: {running_loss:0.6f} , BL{loss.item() / label_front.shape[0]: 0.6f})')
        
    writer.add_scalar('validation loss',
                        running_loss ,
                        epoch )
    writer.close()
    if history is not None:
            history.loc[epoch, 'val_loss'] = running_loss
        
    return running_loss
        

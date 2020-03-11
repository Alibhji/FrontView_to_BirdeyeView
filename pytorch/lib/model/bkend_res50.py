import torch.nn as nn
import torch
from  torchvision import models


class Bkend_res50_8top(nn.Module):
    def __init__(self):
        super(Bkend_res50_8top, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
        
        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.encode_front_bbox =  nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25),            
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True)           
         )
        
        self.decode_to_birdeye_bbox =  nn.Sequential(
            torch.nn.Linear(256+2048, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25), 
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25), 
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25), 
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25),
            
            torch.nn.Linear(128, 4),
            torch.nn.Tanh(),
           
         )
#         self.classifier = nn.Linear(4, 2)
#         self.la
        
    def forward(self, croped_f_image , f_bbox):
        batch = croped_f_image.shape[0]
        x1 = self.resnet50(croped_f_image)
        x1 = x1.view(batch, 2048)
        
        encoded_f_bbox = self.encode_front_bbox(f_bbox)
        x1 = torch.cat((x1, encoded_f_bbox), dim=1)
        out = self.decode_to_birdeye_bbox (x1)
        
        
        
#         x1 = self.norm_layer(x1)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.classifier(F.relu(x))
        return out

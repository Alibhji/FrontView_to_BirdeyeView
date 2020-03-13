import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm


class Dataset_top_to_birdView(Dataset):
    def __init__(self, root , type_='train', dim=(224, 224) ,n_channels= 3 , check_images = False ):
        self.root_dir = root
        
        self.dim = dim
        self.n_channels = n_channels
  
        pd_file = pd.DataFrame()

        print(f"creating a pandas file from {type_} images and their annotations...")
        
        for dir_ in tqdm(root):
            pd_ = self._generate_pandas_file(dir_)
            pd_file = pd_file.append(pd_, ignore_index=True)

        if(check_images):
            print(f"Validating {type_} images pathes ...")            
            non_exitence = self.check_image_existence(pd_file)
            pd_file.drop (non_exitence , inplace=True)
        self.pd_file = pd_file
        self.imgs_f  = np.empty(( *self.dim, self.n_channels))
        self. imgs_b  = np.empty(( *self.dim, self.n_channels))
        self.bboxs_f  = np.empty((4))
        self.bboxs_b  = np.empty((4)) 
            
    def _generate_pandas_file(self , root):
        dataset = dict()
        text_file = os.path.join(root , 'filtered_data.txt' )

        with open (text_file,'r') as file:
            text_file = file.readlines()
            for idx ,line in enumerate(text_file):
                line = line.strip('\n').strip().split(',')
                temp = dict()
                temp['frame_f'] = line[0].strip()
                temp['frame_b'] = line[1].strip()
                temp['bbox_id'] = line[2]
                temp['bbox_model'] = line[3]
                temp['bbox_dist'] = line[4]
                temp['bbox_yaw'] = line[5]
                temp['xf_min'] = line[6]
                temp['yf_min'] = line[7]
                temp['xf_max'] = line[8]
                temp['yf_max'] = line[9]

                temp['xb_min'] = line[10]
                temp['yb_min'] = line[11]
                temp['xb_max'] = line[12]
                temp['yb_max'] = line[13]   
                temp['img_path'] = root
                dataset.update({idx :temp})

        dataset = pd.DataFrame(dataset).T
        
        return dataset
    
    def check_image_existence(self , pd_file):
        counter = 0 
        non_exitence_indexes =[]
        
        for row in pd_file.iterrows():
#             print(row[1])
            row =row[1]
        
            imgf = os.path.join(row.img_path , 'frames' ,row.frame_f)    
            imgb = os.path.join(row.img_path , 'frames' ,row.frame_b)

            if(not os.path.exists(imgf)):
#                 print('****', imgf , row.name)
                counter += 1
                non_exitence_indexes.append(row.name)
        print( 'number of removed labels: ' , counter)
        return non_exitence_indexes
    
    
    def extract_crop(self ,image ,x_min, y_min, x_max, y_max):
        

        h, w = image.shape[:2]
        
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        crop = image[y_min:y_max, x_min:x_max, :].copy()

        crop = cv2.resize(crop, (224, 224))
    
        return crop

    def _generate_crop_imgs(self, index):
        
        imgs_f  = np.empty(( *self.dim, self.n_channels))
        imgs_b  = np.empty(( *self.dim, self.n_channels))
        bboxs_f  = np.empty((4))
        bboxs_b  = np.empty((4))
        
        
        row = self.pd_file.iloc[index]            
            
        imgf = os.path.join(row.img_path , 'frames' ,row.frame_f)    
        imgb = os.path.join(row.img_path  , 'frames' ,row.frame_b)
#             imgf = os.path.join(self.root_dir , 'frames' ,row.frame_f)    
#             imgb = os.path.join(self.root_dir , 'frames' ,row.frame_b) 
            
        bbox_f   = np.array([row.xf_min, row.yf_min  ,row.xf_max ,row.yf_max ],dtype=np.float32)
        bbox_b   = np.array([row.xb_min  , row.yb_min  ,row.xb_max  ,row.yb_max ],dtype=np.float32)
            
        imgf = cv2.imread(imgf , cv2.IMREAD_COLOR)
        imgb = cv2.imread(imgb , cv2.IMREAD_COLOR)
#             if(imgb is None or imgb is None):
#                 print('----------------->', row)
            
        imgs_f  = self.extract_crop(imgf , *tuple(bbox_f) )
        imgs_b  = self.extract_crop(imgb , *tuple(bbox_b) )
        bboxs_f = bbox_f
        bboxs_b = bbox_b
            
        return imgs_f ,imgs_b , bboxs_f ,bboxs_b 
    
    
    def __len__(self):
        #return len(self.images)
        return len(self.pd_file) 
    
    def __getitem__(self, index):
#         print(self.pd_file.iloc[index])
        self.imgs_f ,self.imgs_b , self.bboxs_f ,self.bboxs_b = self._generate_crop_imgs(index)
        self.imgs_f = np.moveaxis(self.imgs_f, 2, 0).astype(np.float32)
        self.imgs_b = np.moveaxis(self.imgs_b, 2, 0).astype(np.float32)
        self.imgs_f =  self.imgs_f / 255.
        self.imgs_b =  self.imgs_b / 255.



	
        return self.bboxs_f , self.imgs_f , self.bboxs_b

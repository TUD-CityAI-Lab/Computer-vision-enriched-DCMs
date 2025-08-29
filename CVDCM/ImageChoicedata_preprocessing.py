import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
from transformers import AutoImageProcessor
from transformers.utils.logging import set_verbosity_error

# Set verbosity of transformers to error
set_verbosity_error()

# Load image processor
image_processor = AutoImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
image_processor.size['height'] = 384
image_processor.size['width'] = 384
image_processor.do_center_crop = False

# Create class for dataset
class ImageChoiceDataset(Dataset):

    def __init__(self, data_file, img_path, set_type, transform=True):
        if set_type == 'train':
            self.annotations = pd.read_csv(data_file).query('train == 1').reset_index(drop=True)
        elif set_type == 'test':
            self.annotations = pd.read_csv(data_file).query('test == 1').reset_index(drop=True)

        # self.annotations = pd.read_csv(data_file)
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        # Get image paths
        img_path1 = os.path.join(self.img_path, self.annotations.loc[index, 'IMG1']) # name first image in first column
        img_path2 = os.path.join(self.img_path, self.annotations.loc[index, 'IMG2']) # name second images in second column
        
        # Load image data
        image1 = io.imread(img_path1)
        image2 = io.imread(img_path2)
        
        # Ensures all image have 3 channels: RGB
        if image1.ndim==2:
            image1 = np.atleast_3d(image1)
            image1 = np.tile(image1,3)

        if image2.ndim==2:
            image2 = np.atleast_3d(image2)
            image2 = np.tile(image2,3)
        
        y_label  = torch.tensor(int( (self.annotations.loc[index, 'CHOICE']-1)==0)) # choice
        tl1       = torch.tensor(float(self.annotations.loc[index, 'TL1']/3))       # price first image
        tl2       = torch.tensor(float(self.annotations.loc[index, 'TL2']/3))       # price second image
        tt1      = torch.tensor(float(self.annotations.loc[index, 'TT1']/10))       # travel time first image
        tt2      = torch.tensor(float(self.annotations.loc[index, 'TT2']/10))       # travel time second image
        img_name1 = self.annotations.loc[index, 'IMG1']
        img_name2 = self.annotations.loc[index, 'IMG2']
        ID = self.annotations.loc[index, 'ID']
               
        # Transform images          
        if self.transform is not None:
            image1 = image_processor(image1, return_tensors = "pt").pixel_values[0]
            image2 = image_processor(image2, return_tensors = "pt").pixel_values[0]

            # if image1.shape[0]==1:
            #     image1 = image1.repeat(3,1,1)

            # if image2.shape[0]==1:
            #     image2 = image2.repeat(3,1,1)
                
        return [image1, image2, y_label, tl1, tl2, tt1, tt2,img_name1,img_name2, ID]

def data_to_cuda(image1, image2, tl1, tl2, tt1, tt2, y_label, device):
        
    # Transfer cost and travel time to cuda
    tl1 = tl1.to(device=device)
    tl2 = tl2.to(device=device)
    tt1 = tt1.to(device=device)
    tt2 = tt2.to(device=device)
        
    # Transfer data to cuda
    image1 = image1.to(device=device)
    image2 = image2.to(device=device)

    y_label = y_label.unsqueeze(1)
    y_label = y_label.float()
    y_label = y_label.to(device=device)
    return image1, image2, tl1, tl2, tt1, tt2, y_label
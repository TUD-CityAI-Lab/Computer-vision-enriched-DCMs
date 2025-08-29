import torch
import torch.nn as nn


class cvdcm_model(nn.Module):

    def __init__(self, path_pretrained_model=None):
        super(cvdcm_model, self).__init__()
        
        # Define the CNN
        self.cvmodel = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained=True)
        
        # Define the linear-additive utility RUM for the feature map
        self.dcm_f = nn.Linear(1000,1, bias = False)
        self.dcm_f.weight.data.normal_(mean=0.0, std=0.1)
        
        
        # Define the linear-additive utility RUM for the monthly housing cost,commute travel time, and months
        self.dcm_p = nn.Linear(2,1,bias=False)
        self.dcm_p.weight = torch.nn.Parameter(self.dcm_p.weight.to(torch.float32))

        # Starting values for traffic lights and cycling time
        self.dcm_p.weight.data[0][0].fill_(-0.70) # beta_tl
        self.dcm_p.weight.data[0][1].fill_(-0.95) # beta_tt

        if path_pretrained_model is not None:
            self.load_state_dict(torch.load(path_pretrained_model), strict=True)

    
    def forward_once(self, image,tl,tt):
        
        # Extract features from image
        feature_map = self.cvmodel(image)            
        feature_map = torch.squeeze(feature_map)
          
        # Apply DCM to feature map
        V_f = self.dcm_f(feature_map) 
        
        # Add dimension
        tl = tl[:,None]
        tt = tt[:,None]
        
        # Concatenate
        x = torch.cat((tl,tt),1).to(torch.float32)
        
        # Utility from numeric attributes
        V_dcm = self.dcm_p(x).to(torch.float)
        
        # Total utility
        V = V_f + V_dcm

        return V
    
    def forward(self, image1, image2, tl1, tl2, tt1, tt2):
        V1 = self.forward_once(image1,tl1,tt1)
        V2 = self.forward_once(image2,tl2,tt2)

        # Return probability
        prob = torch.div(1,torch.add(torch.exp(torch.subtract(V2, V1)),1))   # out = probability of alt1 (left)

        # Replace NaN with smallest possible float in torch
        prob = torch.nan_to_num(prob, nan=torch.finfo(torch.float32).tiny)

        # Return outputs
        return [prob, V1, V2]

    def return_featuremap(self, image):
        
        # Extract features from image
        feature_map = self.cvmodel(image)            
        feature_map = torch.squeeze(feature_map)

        # Apply DCM to features
        V_f = self.dcm_f(feature_map) 
    
        return feature_map, V_f





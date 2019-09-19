import torch
from torchvision import transforms


from .models import SuperResolver, SuperRes4x, SuperResST, SuperResST8, Generator

superres = SuperResST8()
state_dict = torch.load('project/superresST8_perc_3500.pth', map_location='cpu')
superres.load_state_dict(state_dict)
'''
superres = Generator(4)
state_dict = torch.load('project/srgan_2000.pth', map_location='cpu')
superres.load_state_dict(state_dict)
'''

def superresolve(img, res=4, seed=2019):
    return superres(img)

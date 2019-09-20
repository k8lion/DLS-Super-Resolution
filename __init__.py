import torch
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from .models import SuperResolver, SuperRes4x, SuperResST, SuperResST8, Generator, RRDBNet


superres0 = SuperResST8()
state_dict = torch.load('project/superresST8_perc_3500.pth', map_location='cpu')
superres0.load_state_dict(state_dict)

superres1 = Generator(4)
state_dict = torch.load('project/srgan_5500.pth', map_location='cpu')
superres1.load_state_dict(state_dict)


superres2 = RRDBNet(3, 3, 64, 23, gc=32)
superres2.load_state_dict(torch.load('project/RRDB_ESRGAN_x4.pth'), strict=True)


def superresolve(img, seed=2019):
    x = transforms.ToTensor()(img).unsqueeze(0)
    '''
    x0 = superres0(x)
    x1 = superres1(x)
    x2 = superres2(x)
    x = torch.mean(torch.stack([x0,x1,x2]), dim=0)
    '''
    summed = list()
    for s in np.arange(0.5, 2.0, 0.5):
        scaled = F.interpolate(x, scale_factor=2**s, mode='bicubic', align_corners=False)
        x2 = superres2(scaled)
        out = F.interpolate(x2, size=256, mode='bicubic', align_corners=False)
        summed.append(out)
    x = torch.mean(torch.stack(summed), dim=0)
    return transforms.ToPILImage()(x.select(0,0))

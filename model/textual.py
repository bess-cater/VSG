#here feature pyramid!!
import torch
import torch.nn.functional as F
from torch import nn
from config import config


#TODO think about maxpool operation!
#TODO Need to add padding!!! 
class Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottom = nn.Conv1d(300, 450, 2, stride=1)
        self.middle = nn.Conv1d(config.MODEL.PYRAMID.chann1, 700, 2, stride=2)
        self.top = nn.Conv1d(config.MODEL.PYRAMID.chann2, 300, 2, stride=3)
        self.RCNN_toplayer= nn.Conv1d(config.MODEL.PYRAMID.chann3, 300, kernel_size=1, stride=1)
        self.RCNN_latlayer1 = nn.Conv1d(config.MODEL.PYRAMID.chann2, 300, kernel_size=1, stride=1)
        self.RCNN_latlayer2 = nn.Conv1d(config.MODEL.PYRAMID.chann1, 300, kernel_size=1, stride=1)
        self.RCNN_smooth1=nn.Conv1d(300, 300, kernel_size=2, stride=1)
        self.RCNN_smooth2=nn.Conv1d(300, 300, kernel_size=2, stride=1)

    def forward(self, text):
       #? Only for input not in batch, adding 1st dimension!
       x = text[None, :, :]
       x = x.permute(0,2,1) 
       c1 = self.bottom(x)
       c2 = self.middle(c1)
       c3 = self.top(c2)
       p5 = self.RCNN_toplayer(c3) #connected with first video input
       p4 = self._upsample_add(p5, self.RCNN_latlayer1(c2))
       p4 = self.RCNN_smooth1(p4) #connected with second video input

       p3 = self._upsample_add(p4, self.RCNN_latlayer2(c1))
       p3 = self.RCNN_smooth2(p3) #connected with third video inpu
       return p5, p4, p3
    def _upsample_add(self, x, y):
        """in original paper use NNeighbour, in code for some reason = Bilinear.
        See https://github.com/jwyang/fpn.pytorch/blob/master/lib/model/fpn/fpn.py#L99C10-L99C10
        x = top; y = lateral!  
        BUT with 3d (not 4d) input bilinear impossible --> nearest"""

        _,_,W = y.size()
        return nn.functional.interpolate(x, size=W, mode='nearest') + y
    
text = torch.randint(1,5, (10, 300), dtype=torch.float)
head = Pyramid()
a, b, c = head(text)
for i in a,b,c:
    print(i.shape)
    """
torch.Size([1, 300, 1])
torch.Size([1, 300, 3])
torch.Size([1, 300, 8])
      |
    |  |  |
| |  |  |  | |

"""



import torch
from torch import nn
import torch.nn.functional as F


def nmse(pred,y):
    return (pred-y).square().sum((1,2,3))/y.square().sum((1,2,3)).clamp_min(1e-12)


class Block(nn.Module):
    def __init__(self,inc,out):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(inc,out,3,padding=1),nn.GroupNorm(8,out),nn.SiLU(),
                               nn.Conv2d(out,out,3,padding=1),nn.GroupNorm(8,out),nn.SiLU())
    def forward(self,x):return self.net(x)


class Reconstructor(nn.Module):
    def __init__(self,bridge,hybrid=True,width=32):
        super().__init__()
        self.bridge=bridge;self.hybrid=hybrid
        self.enc1=Block(37,width);self.enc2=Block(width,width*2);self.middle=Block(width*2,width*4)
        self.dec2=Block(width*6,width*2);self.dec1=Block(width*3,width)
        self.out=nn.Conv2d(width,4,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,variance,scale):
        base=self.bridge(x,variance) if self.hybrid else torch.zeros_like(x[:,:4])
        noise=torch.log10((variance.mean(1)/scale.square()).clamp_min(1e-12))[:,None,None,None].expand(-1,1,25,25)
        a=self.enc1(torch.cat((x,base,noise),1))
        b=self.enc2(F.avg_pool2d(a,2));c=self.middle(F.avg_pool2d(b,2))
        d=self.dec2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e=self.dec1(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return base+self.out(e)

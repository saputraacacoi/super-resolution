import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, nin, nout, ks=3, stride=1, padding=1, use_relu=True):
        super(ResidualBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(nin)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nin, nout, kernel_size=ks, stride=stride, padding=padding)
        self.bn2   = nn.BatchNorm2d(nout)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nout, nout, kernel_size=ks, stride=stride, padding=padding)
        self.use_relu = use_relu
        self.nin = nin
        self.nout = nout
        
    def forward(self, xin):
        x = self.bn1(xin)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = torch.add(xin,x)
        return x

class BaseResConv2D(nn.Module):
    def __init__(self, nin, nout, ks=3, stride=1, padding=1):
        super(BaseResConv2D, self).__init__()
        self.conv2d = nn.Conv2d(nin, nout, kernel_size=ks, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.conv2d(x)
        return x
    
class FirstBlock(nn.Module):
    def __init__(self, nin, nout, pool='stride'):
        super(FirstBlock, self).__init__()
        self.baseconv  = BaseResConv2D(nin, nout)
        self.resblock1 = ResidualBlock(nout, nout)
        self.resblock2 = ResidualBlock(nout, nout)
        self.maxpool2d = nn.MaxPool2d(2)
        self.cstride   = nn.Conv2d(nout, nout, kernel_size=3, stride=2, padding=1)
        self.pool = pool
        
        
    def forward(self, x):
        c1 = self.baseconv(x)
        r1 = self.resblock1(c1)
        r2 = self.resblock2(r1)
        link = torch.add(c1, r2)
        
        if self.pool == 'stride':
            pool = self.cstride(link)
        else:
            pool = self.maxpool2d(link)
            
        return pool, link

class DownBlock(nn.Module):
    def __init__(self, nin, nout, pool='stride'):
        super(DownBlock, self).__init__()
        self.baseconv  = BaseResConv2D(nin, nout)
        self.resblock1 = ResidualBlock(nout, nout)
        self.resblock2 = ResidualBlock(nout, nout)
        self.maxpool2d = nn.MaxPool2d(2)
        self.cstride   = nn.Conv2d(nout, nout, kernel_size=3, stride=2, padding=1)
        self.pool = pool
        
    def forward(self, x):
        c1 = self.baseconv(x)
        r1 = self.resblock1(c1)
        r2 = self.resblock2(r1)
        link = torch.add(c1, r2)
        
        if self.pool == 'stride':
            pool = self.cstride(link)
        else:
            pool = self.maxpool2d(link)
            
        return pool, link

class CenterBlock(nn.Module):
    def __init__(self, nin, nout):
        super(CenterBlock, self).__init__()
        self.resblock1 = ResidualBlock(nin, nout)
        self.resblock2 = ResidualBlock(nout, nout)
        self.resblock3 = ResidualBlock(nout, nout)
    
    def forward(self, x):
        rb1 = self.resblock1(x)
        rb2 = self.resblock2(rb1)
        rb3 = self.resblock3(rb2)
        
        out = torch.add(x,rb3)
        return out

    
class UpBlock(nn.Module):
    def __init__(self, nin, nout, upsample='transpose'):
        super(UpBlock, self).__init__()
        self.upsample2d = nn.Upsample(scale_factor=2, mode='nearest')
        self.ctranspose = nn.ConvTranspose2d(nin, nin, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.baseconv   = BaseResConv2D(nin,nout)
        self.resblock1  = ResidualBlock(nout, nout)
        self.resblock2  = ResidualBlock(nout, nout)
        self.resblock3  = ResidualBlock(nout, nout)
       
       
        self.upsample = upsample
        
    def forward(self, x, link):
        if self.upsample == 'transpose':
            x = self.ctranspose(x)
        else:
            x = self.upsample2d(x)
        x = torch.add(x,link)
        
        x = self.baseconv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        
        return x
    
    
class LastBlock(nn.Module):
    def __init__(self, nin, nout, upsample='transpose'):
        super(LastBlock, self).__init__()
        self.upsample2d = nn.Upsample(scale_factor=2, mode='nearest')
        self.ctranspose = nn.ConvTranspose2d(nin, nin, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.baseconv1  = BaseResConv2D(nin, nin)
        self.resblock1  = ResidualBlock(nin, nin)
        self.resblock2  = ResidualBlock(nin, nin)
        self.baseconv2  = BaseResConv2D(nin, nout)
        
        self.sigmoid = nn.Sigmoid()
        self.upsample = upsample
    
    def forward(self, x, link):
        if self.upsample == 'transpose':
            x = self.ctranspose(x)
        else:
            x = self.upsample2d(x)
        x = torch.add(x,link)
        
        x = self.baseconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.baseconv2(x)
        x = self.sigmoid(x)
        return x
    
    
class UResNet(nn.Module):
    def __init__(self, ngpu, nin, nout, nstart=32):
        super(UResNet, self).__init__()
        self.ngpu = ngpu
        
        self.input = FirstBlock(nin, nstart)
        
        self.down1 = DownBlock(nstart, nstart*2)
        self.down2 = DownBlock(nstart*2, nstart*4)
        self.down3 = DownBlock(nstart*4, nstart*8)
        
        self.center = CenterBlock(nstart*8, nstart*8)
        
        self.up3 = UpBlock(nstart*8, nstart*4)
        self.up2 = UpBlock(nstart*4, nstart*2)
        self.up1 = UpBlock(nstart*2, nstart)
        
        self.output = LastBlock(nstart, nout)
        
    def forward(self, x):
        inp, flink = self.input(x)
        
        d1, link1 = self.down1(inp)
        d2, link2 = self.down2(d1)
        d3, link3 = self.down3(d2)
        
        c = self.center(d3)
        
        u3 = self.up3(c, link3)
        u2 = self.up2(u3, link2)
        u1 = self.up1(u2, link1)
        
        out = self.output(u1, flink)
        
        return out
    
    
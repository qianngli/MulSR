import torch
import torch.nn as nn
import pdb                                    


#=================================ALL=================================        
class Unit(nn.Module):
    def __init__(self, wn, n_feats, kernel_size = 3, padding = 1, bias = True, act=nn.ReLU(inplace=True)):
        super(Unit, self).__init__()        
        
        m = []
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))) 
        m.append(act)
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))) 
   
        self.m = nn.Sequential(*m)
    
    def forward(self, x):
        
        out = self.m(x) + x   

        return out 

class CrossFusion(nn.Module):
    def __init__(self, wn, n_feats):
        super(CrossFusion, self).__init__()

        conv3D = []
        conv3D.append(wn(nn.Conv3d(48, 48, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        conv3D.append(wn(nn.Conv3d(48, 48, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))        
        self.conv3D = nn.Sequential(*conv3D)
    
        self.conv_diff = wn(nn.Conv2d(n_feats, n_feats+8, kernel_size=3, padding=1, bias=True)) 
        self.reduce_D = wn(nn.Conv2d(n_feats*2, n_feats+8, kernel_size=3, padding=1, bias=True)) 

        self.gamma = nn.Parameter(torch.ones(3))   

        self.conv =  wn(nn.Conv2d(48*3, n_feats, kernel_size=3, padding=1, bias=True)) 
              
    def forward(self, data):
        diff = self.conv_diff(data[0]-data[1])

        fusion = self.reduce_D(torch.cat([data[0], data[1]], 1))

        x = torch.cat([self.gamma[0]*fusion[:,0:48,:,:].unsqueeze(2), self.gamma[1]*torch.cat([fusion[:,48:,:,:],diff[:,0:24,:,:]],1).unsqueeze(2),
                       self.gamma[2]*diff[:,24:,:,:].unsqueeze(2)], 2)
        x = self.conv3D(x) + x

        x = torch.cat([x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]], 1)
        x = self.conv(x)  

        return x

class convDU(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(9,1)):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.PReLU()
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).reshape(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]

        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(1,9)):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.PReLU()
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).reshape(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]

        fea = torch.cat(fea_stack, 3)
        return fea

class RDE(nn.Module):
    def __init__(self, wn, n_feats):
        super(RDE, self).__init__()

        self.crossFusion = CrossFusion(wn, n_feats) 

        self.rgb_first = Unit(wn, n_feats) 
        self.hsi_first = Unit(wn, n_feats) 

        self.rgb_end = Unit(wn, n_feats) 
        self.hsi_end = Unit(wn, n_feats)

    def forward(self, data):
        hsi = data[0]
        rgb = data[1]
        
        rgb = self.rgb_first(rgb)
        hsi = self.hsi_first(hsi)

        hsi_rgb = self.crossFusion([hsi, rgb])  

        rgb = self.rgb_end(rgb + hsi_rgb)
        hsi = self.hsi_end(hsi + hsi_rgb)

        return hsi + data[0], rgb + data[1]

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.upscale_factor  
        n_feats = args.n_feats          
        kernel_size = 3
        self.n_module = 4

        wn = lambda x: torch.nn.utils.weight_norm(x) 

        self.hsi_head = wn(nn.Conv2d(3, n_feats, kernel_size=kernel_size, padding=1, bias=True)) 
        self.rgb_head = wn(nn.Conv2d(3, n_feats, kernel_size=kernel_size, padding=1, bias=True)) 
        self.nearest = nn.Upsample(scale_factor=int(scale/2), mode='nearest')
        self.nearest_rgb = nn.Upsample(scale_factor=0.5, mode='nearest')

        self.hsi_head_x2 = wn(nn.Conv2d(3, n_feats, kernel_size=kernel_size, padding=1, bias=True)) 
        self.rgb_head_x2 = wn(nn.Conv2d(3, n_feats, kernel_size=kernel_size, padding=1, bias=True)) 
        self.nearest_x2 = nn.Upsample(scale_factor=scale, mode='nearest')

        self.up = nn.Sequential(
            wn(nn.Conv2d(n_feats, int(4*n_feats),  kernel_size=1)),
            nn.PixelShuffle(2)
        )
        self.down = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=2, padding=int(kernel_size/2), bias=True)

        self.optimize_up = nn.Sequential(
            wn(nn.Conv2d(n_feats, int(4*n_feats), kernel_size=1)),
            nn.PixelShuffle(2)
        )
        self.optimize_down = wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=2, padding=int(kernel_size/2), bias=True))

        inter_body = [
                      RDE(wn, n_feats
                  ) for _ in range(self.n_module)
        ]
        self.inter_body =  nn.Sequential(*inter_body) 

        inter_body_x2 = [
                     RDE(wn, n_feats
                 ) for _ in range(self.n_module)
        ]
        self.inter_body_x2 =  nn.Sequential(*inter_body_x2) 

        self.DULR = nn.Sequential(convDU(in_out_channels=64,kernel_size=(1,7)),
                                  convLR(in_out_channels=64,kernel_size=(7,1)))

        self.uint = Unit(wn, n_feats)
        self.uint_x2 = Unit(wn, n_feats)

        self.reduce = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1)) 
        self.reduce_x2 = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1))

        self.cross_reduce = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1))

        self.end_x2 = nn.Sequential(
            wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1, bias=True)),
            wn(nn.Conv2d(n_feats, 1, kernel_size=kernel_size, padding=1, bias=True)) 
        )


    def forward(self, hsi, rgb):

        x = self.nearest(hsi)  
        x = self.hsi_head(x)
        y = self.rgb_head(self.nearest_rgb(rgb))

        x2 = self.nearest_x2(hsi)   
        x2 = self.hsi_head_x2(x2)
        y2 = self.rgb_head_x2(rgb)


        for i in range(self.n_module):        
            x, y = self.inter_body[i]([x, y])
            x2, y2 = self.inter_body_x2[i]([x2, y2])

        x = self.reduce(torch.cat([x, y], 1))

        x2 = self.reduce_x2(torch.cat([x2, y2], 1))

        fusion = self.cross_reduce(torch.cat([self.down(x2), x], 1))
        fusion = self.DULR(fusion)
        x = x + fusion
        fusion = self.up(fusion)
        x2 = x2 + fusion

        x = self.uint(x)
        x2 = self.uint_x2(x2) 
        
        x = self.optimize_down(x2) - x
        x2 = self.optimize_up(x) + x2

        x2 = self.end_x2(x2)        
        
        return x, x2



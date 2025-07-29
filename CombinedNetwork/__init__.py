import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import layers

device = torch.device("cuda:0")

def ConsProj(initial_v):
    [B, C, Nt, Nx, Ny] = initial_v.shape
    proj_v = torch.zeros((B,C,Nt,Nx,Ny))
    for i in range(B):
        for j in range(C):
            mean_v = torch.mean(initial_v[i,j,:,:,:],dim=0,keepdim=True)
            proj_v[i,j,:,:,:] = initial_v[i,j,:,:,:] - mean_v

    return proj_v

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.head_dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.W_query = nn.Linear(dim, dim)
        self.W_key = nn.Linear(dim, dim)
        self.W_value = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)
        self.out = nn.Linear(dim, dim)

    def forward(self,x1,x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        query = self.W_query(x1)
        key = self.W_key(x2)
        value = self.W_value(x2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.attend(scores)
        attention_values = torch.matmul(attention_weights, value)
        return self.out(attention_values)

class MLPBlock(nn.Module):
    def __init__(self, dim):
        super(MLPBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.MLP1 = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.MLP2 = nn.Linear(4 * dim, dim)

    def forward(self,x1):
        x1 = self.norm(x1)
        attention_x1 = self.MLP1(x1)
        attention_x1 = self.gelu(attention_x1)
        attention_x1 = self.MLP2(attention_x1)

        return attention_x1

class TransformerBlock(nn.Module):
    def __init__(self,dim):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(dim)
        self.MLP = MLPBlock(dim)

    def forward(self,x1, x2, B, C, T, W, L, id):
        x1 = x1 + self.attention(x1, x2)
        x1 = x1 + self.MLP(x1)
        if id == 2:
            x1 = rearrange(x1,'(B T) (W L) C -> B C T W L', B=B, T=T, W=W, L=L, C=C)
        if id == 3:
            x1 = rearrange(x1,'(B W L) T C -> B C T W L', B=B, T=T, W=W, L=L, C=C)

        return x1

class AttentionBlock(nn.Module):
    def __init__(self, C, T, W, L):
        super(AttentionBlock, self).__init__()
        self.spacialposembed = nn.Parameter(torch.zeros(1, W*L, C))
        self.temporalposembed = nn.Parameter(torch.zeros(1, T, C))

        self.cross_regis_spacial_attention = TransformerBlock(C)
        self.cross_seg_spacial_attention = TransformerBlock(C)
        self.cross_regis_temporal_attention = TransformerBlock(C)
        self.cross_seg_temporal_attention = TransformerBlock(C)

        self.self_regis_spacial_attention = TransformerBlock(C)
        self.self_seg_spacial_attention = TransformerBlock(C)
        self.self_regis_temporal_attention = TransformerBlock(C)
        self.self_seg_temporal_attention = TransformerBlock(C)

        self.self_regis_spacial_attention2 = TransformerBlock(C)
        self.self_seg_spacial_attention2 = TransformerBlock(C)
        self.self_regis_temporal_attention2 = TransformerBlock(C)
        self.self_seg_temporal_attention2 = TransformerBlock(C)

    def forward(self, x1, x2):
        [B, C, T, W, L] = x1.shape

        rearranged_x1 = rearrange(x1, 'B C T W L -> (B T) (W L) C') + self.spacialposembed
        rearranged_x2 = rearrange(x2, 'B C T W L -> (B T) (W L) C') + self.spacialposembed
        x1 = self.self_regis_spacial_attention(rearranged_x1, rearranged_x1, B, C, T, W, L, 2)
        x2 = self.self_seg_spacial_attention(rearranged_x2, rearranged_x2, B, C, T, W, L, 2)

        rearranged_x1 = rearrange(x1, 'B C T W L -> (B W L) T C') + self.temporalposembed
        rearranged_x2 = rearrange(x2, 'B C T W L -> (B W L) T C') + self.temporalposembed
        x1 = self.self_regis_temporal_attention(rearranged_x1, rearranged_x1, B, C, T, W, L, 3)
        x2 = self.self_seg_temporal_attention(rearranged_x2, rearranged_x2, B, C, T, W, L, 3)

        rearranged_spacial_x1 = rearrange(x1, 'B C T W L -> (B T) (W L) C')
        rearranged_spacial_x2 = rearrange(x2, 'B C T W L -> (B T) (W L) C')
        cross_spacialattention_output1 = self.cross_regis_spacial_attention(rearranged_spacial_x1, rearranged_spacial_x2, B, C, T, W, L, 2)
        cross_spacialattention_output2 = self.cross_seg_spacial_attention(rearranged_spacial_x2, rearranged_spacial_x1, B, C, T, W, L, 2)

        rearranged_temporal_x1 = rearrange(x1, 'B C T W L -> (B W L) T C')
        rearranged_temporal_x2 = rearrange(cross_spacialattention_output1, 'B C T W L -> (B W L) T C')
        x1 = self.cross_regis_temporal_attention(rearranged_temporal_x1, rearranged_temporal_x2, B, C, T, W, L, 3)

        rearranged_temporal_x3 = rearrange(x2, 'B C T W L -> (B W L) T C')
        rearranged_temporal_x4 = rearrange(cross_spacialattention_output2, 'B C T W L -> (B W L) T C')
        x2 = self.cross_seg_temporal_attention(rearranged_temporal_x3, rearranged_temporal_x4, B, C, T, W, L, 3)

        rearranged_x1 = rearrange(x1, 'B C T W L -> (B T) (W L) C')
        rearranged_x2 = rearrange(x2, 'B C T W L -> (B T) (W L) C')
        x1 = self.self_regis_spacial_attention2(rearranged_x1, rearranged_x1, B, C, T, W, L, 2)
        x2 = self.self_seg_spacial_attention2(rearranged_x2, rearranged_x2, B, C, T, W, L, 2)

        rearranged_x1 = rearrange(x1, 'B C T W L -> (B W L) T C')
        rearranged_x2 = rearrange(x2, 'B C T W L -> (B W L) T C')
        x1 = self.self_regis_temporal_attention2(rearranged_x1, rearranged_x1, B, C, T, W, L, 3)
        x2 = self.self_seg_temporal_attention2(rearranged_x2, rearranged_x2, B, C, T, W, L, 3)
        # [B, C, T, W, L]

        return [x1, x2]

def build_conv_block(in_c, out_c1, out_c2):
    conv_block = []
    conv_block += [
        nn.Conv3d(in_channels=in_c, out_channels=out_c1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                  padding_mode='zeros')]
    conv_block += [
        nn.Conv3d(in_channels=out_c1, out_channels=out_c1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                  padding_mode='circular')]
    conv_block += [nn.GroupNorm(16,out_c1)]
    conv_block += [nn.ReLU()]
    conv_block += [
        nn.Conv3d(in_channels=out_c1, out_channels=out_c2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                  padding_mode='zeros')]
    conv_block += [
        nn.Conv3d(in_channels=out_c2, out_channels=out_c2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                  padding_mode='circular')]
    conv_block += [nn.GroupNorm(16,out_c2)]
    conv_block += [nn.ReLU()]

    return nn.Sequential(*conv_block)

class CombinedNet(nn.Module):

    def __init__(self,inshape, joint_regis = True, add_attention = True, size = 16):
        super(CombinedNet, self).__init__()
        self.joint_regis = joint_regis
        self.add_attention = add_attention

        self.Regisconv1 = build_conv_block(1, size, size)
        self.Regisconv2 = build_conv_block(size, size*2, size*2)
        self.Regisconv3 = build_conv_block(size*2, size*4, size*4)
        self.Regisconv4 = build_conv_block(size*4, size*8, size*4)
        if joint_regis:
            self.Segconv1 = build_conv_block(1, size, size)
            self.Segconv2 = build_conv_block(size, size*2, size*2)
            self.Segconv3 = build_conv_block(size*2, size*4, size*4)
            self.Segconv4 = build_conv_block(size*4, size*8, size*4)

        self.Regisconv5 = build_conv_block(size*8, size*4, size*2)
        self.Regisconv6 = build_conv_block(size*4, size*2, size)
        self.Regisconv7 = build_conv_block(size*2, size, size)
        if joint_regis:
            self.Segconv5 = build_conv_block(size*8, size*4, size*2)
            self.Segconv6 = build_conv_block(size*4, size*2, size)
            self.Segconv7 = build_conv_block(size*2, size, size)
        
        self.Regisconv8 = nn.Sequential(nn.Conv3d(in_channels=size, out_channels=2, kernel_size=1, stride=1, padding=0))
        if joint_regis:
            self.Segconv8 = nn.Sequential(nn.Conv3d(in_channels=size, out_channels=2, kernel_size=1, stride=1, padding=0))

        # downsample layer
        self.RegisdownSample = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        if joint_regis:
            self.SegdownSample = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        # upsample layer
        self.RegisUpSample = nn.Upsample(scale_factor=(1, 2, 2),mode='trilinear',align_corners=True)
        if joint_regis:
            self.SegUpSample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        self.integrate = layers.VecInt(inshape, 7)
        self.transformer = layers.SpatialTransformer(inshape)

        if add_attention:
            self.attentionblock = AttentionBlock(64, 25, 20, 20)

    def forward(self, x):
        x = x.float()
        x = x.permute(0,1,4,2,3)

        RegisC1 = self.Regisconv1(x)
        RegisD1 = self.RegisdownSample(RegisC1)
        RegisC2 = self.Regisconv2(RegisD1)
        RegisD2 = self.RegisdownSample(RegisC2)
        RegisC3 = self.Regisconv3(RegisD2)
        RegisD3 = self.RegisdownSample(RegisC3)
        RegisC4 = self.Regisconv4(RegisD3)

        if self.joint_regis:
            SegC1 = self.Segconv1(x)
            SegD1 = self.SegdownSample(SegC1)
            SegC2 = self.Segconv2(SegD1)
            SegD2 = self.SegdownSample(SegC2)
            SegC3 = self.Segconv3(SegD2)
            SegD3 = self.SegdownSample(SegC3)
            SegC4 = self.Segconv4(SegD3)

        if self.add_attention:
            [RegisC4, SegC4] = self.attentionblock(RegisC4, SegC4)

        RegisU3 = self.RegisUpSample(RegisC4)
        Regisconcat3 = torch.cat((RegisC3, RegisU3), dim=1)
        RegisC5 = self.Regisconv5(Regisconcat3)
        RegisU2 = self.RegisUpSample(RegisC5)
        Regisconcat2 = torch.cat((RegisC2, RegisU2), dim=1)
        RegisC6 = self.Regisconv6(Regisconcat2)
        RegisU1 = self.RegisUpSample(RegisC6)
        Regisconcat1 = torch.cat((RegisC1, RegisU1), dim=1)
        RegisC7 = self.Regisconv7(Regisconcat1)
        Regisoutput = self.Regisconv8(RegisC7)

        if self.joint_regis:
            SegU3 = self.SegUpSample(SegC4)
            Segconcat3 = torch.cat((SegC3, SegU3), dim=1)
            SegC5 = self.Segconv5(Segconcat3)
            SegU2 = self.SegUpSample(SegC5)
            Segconcat2 = torch.cat((SegC2, SegU2), dim=1)
            SegC6 = self.Segconv6(Segconcat2)
            SegU1 = self.SegUpSample(SegC6)
            Segconcat1 = torch.cat((SegC1, SegU1), dim=1)
            SegC7 = self.Segconv7(Segconcat1)
            Segoutput = self.Segconv8(SegC7)

        InitialV = ConsProj(Regisoutput).to(device)
        InitialNegV = -InitialV.to(device)
        V = self.integrate(InitialV)
        NegV = self.integrate(InitialNegV)
        PosImage = self.transformer(x,V)

        if self.joint_regis:
            PosMask = self.transformer(Segoutput, V)
            PosMask = F.softmax(PosMask, dim=1)
            Segoutput = F.softmax(Segoutput,dim=1)
            return [Segoutput, PosMask, PosImage, NegV, V]
        else:
            return [PosImage, NegV, V]
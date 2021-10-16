# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:34:43 2021

@author: user
"""
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

#class ViT(nn.Module):
#definit(self,*,image_size,patch_size, num_classes,dim,
#  depth,heads,mlp_dim,pool='cls', channels = 3, dim_head = 64,dropout=0.,emb_dropout=0.):
#v = ViT(image_size =256,patch_size = 32, num_classes = 1000,dim = 1024,
#  depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1,  emb_dropout = 0.1)     
image_size =256  # the maximum of the width and height
patch_size = 32 #n = (image_size // patch_size) ** 2   # >16
num_classes = 1000
dim = 1024 #Last dimension of output tensor after linear transformation
depth = 6  #Number of Transformer blocks
heads = 16 #Number of heads in Multi-head Attention layer.
mlp_dim = 2048  #Dimension of the MLP (FeedForward) layer.
pool='cls' 
channels = 3  #default 3. Number of image's channels.
dim_head = 64
dropout = 0.1  #float between [0, 1], default 0.. Dropout rate.
emb_dropout = 0.1  #float between [0, 1], default 0. Embedding dropout rate.

##def forward(self, img):
img = torch.randn(1, 3, 256, 256)
##x = self.to_patch_embedding(img)
image_height, image_width = pair(image_size)
patch_height, patch_width = pair(patch_size)
assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

num_patches = (image_height // patch_height) * (image_width // patch_width)
patch_dim = channels * patch_height * patch_width

#self.to_patch_embedding = nn.Sequential(
# Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
# nn.Linear(patch_dim, dim),)
x =  nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
    nn.Linear(patch_dim, dim),)(img)
#x1=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)(img)
#x2=nn.Linear(patch_dim, dim)(x1)
b, n, _ = x.shape #1, 64
##cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) #(1, 65, 1024)
cls_token = nn.Parameter(torch.randn(1, 1, dim)) #(1,1,1024)
dropoutF = nn.Dropout(emb_dropout)
cls_tokens = repeat(cls_token, '() n d -> b n d', b = b)
x = torch.cat((cls_tokens, x), dim=1) #(1,1,1024) (1, 64, 1024) ->(1,65, 1024)
##x += self.pos_embedding[:, :(n + 1)]# +(1,65,1024)
x += pos_embedding[:, :(n + 1)] #(1,65,1024)
x =dropoutF(x)  #(1,65,1024)
##x = self.transformer(x)
##self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#def_init_(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
layers = nn.ModuleList([])

   
#class PreNorm(nn.Module):  def __init__(self, dim, fn):
norm = nn.LayerNorm(dim) #fn = fn
#def forward(self, x, **kwargs):return self.fn(self.norm(x), **kwargs)

#class Attention(nn.Module):def_init_(self,dim, heads=8, dim_head = 64, dropout =0.):
#def forward(self, x):
#qkv = self.to_qkv(x).chunk(3, dim = -1)
inner_dim = dim_head *  heads #64*16 =1024
to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #1024->3072
attend = nn.Softmax(dim = -1)
scale = dim_head ** -0.5
project_out = not (heads == 1 and dim_head == dim)
to_out = nn.Sequential(
    nn.Linear(inner_dim, dim),
    nn.Dropout(dropout)
) if project_out else nn.Identity()

#class FeedForward(nn.Module):definit(self, dim, hidden_dim, dropout = 0.):
hidden_dim=mlp_dim
net = nn.Sequential(
    nn.Linear(dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, dim),
    nn.Dropout(dropout)
)

#1
#layers.append(nn.ModuleList([
    #PreNorm(dim, Attention(dim, heads=heads, dim_head =dim_head, dropout=dropout)),
x=norm(x)
qkv =to_qkv(x).chunk(3, dim = -1) #(1,65,1024) *3
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv) #(1,16,65,64) *3

dots=torch.matmul(q, k.transpose(-1,-2))*scale #(1,16,65,64) (1,16,64,65)->(1,16,65,65)
attn = attend(dots) #(1,16,65,65)
out = torch.matmul(attn, v) #(1,16,65,65) (1,16,65,64)->(1,16,65,64)
out = rearrange(out, 'b h n d -> b n (h d)') #(1,65, 1024)
Atten_Out_1 =to_out(out)
x = Atten_Out_1 +x
    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
x=norm(x)
FeedForward_Out_1=net(x)
x = FeedForward_Out_1 +x
#]))

#2
#layers.append(nn.ModuleList([
    #PreNorm(dim, Attention(dim, heads=heads, dim_head =dim_head, dropout=dropout)),
x=norm(x)
qkv =to_qkv(x).chunk(3, dim = -1) #(1,65,1024) *3
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv) #(1,16,65,64) *3

dots=torch.matmul(q, k.transpose(-1,-2))*scale #(1,16,65,64) (1,16,64,65)->(1,16,65,65)
attn = attend(dots) #(1,16,65,65)
out = torch.matmul(attn, v) #(1,16,65,65) (1,16,65,64)->(1,16,65,64)
out = rearrange(out, 'b h n d -> b n (h d)') #(1,65, 1024)
Atten_Out_2 =to_out(out)
x = Atten_Out_2 +x
    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
x=norm(x)
FeedForward_Out_2=net(x)
x = FeedForward_Out_2 +x
#]))

#3
#layers.append(nn.ModuleList([
    #PreNorm(dim, Attention(dim, heads=heads, dim_head =dim_head, dropout=dropout)),
x=norm(x)
qkv =to_qkv(x).chunk(3, dim = -1) #(1,65,1024) *3
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv) #(1,16,65,64) *3

dots=torch.matmul(q, k.transpose(-1,-2))*scale #(1,16,65,64) (1,16,64,65)->(1,16,65,65)
attn = attend(dots) #(1,16,65,65)
out = torch.matmul(attn, v) #(1,16,65,65) (1,16,65,64)->(1,16,65,64)
out = rearrange(out, 'b h n d -> b n (h d)') #(1,65, 1024)
Atten_Out_3 =to_out(out)
x = Atten_Out_3 +x
    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
x=norm(x)
FeedForward_Out_3=net(x)
x = FeedForward_Out_3 +x
#]))

#4
#layers.append(nn.ModuleList([
    #PreNorm(dim, Attention(dim, heads=heads, dim_head =dim_head, dropout=dropout)),
x=norm(x)
qkv =to_qkv(x).chunk(3, dim = -1) #(1,65,1024) *3
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv) #(1,16,65,64) *3

dots=torch.matmul(q, k.transpose(-1,-2))*scale #(1,16,65,64) (1,16,64,65)->(1,16,65,65)
attn = attend(dots) #(1,16,65,65)
out = torch.matmul(attn, v) #(1,16,65,65) (1,16,65,64)->(1,16,65,64)
out = rearrange(out, 'b h n d -> b n (h d)') #(1,65, 1024)
Atten_Out_4 =to_out(out)
x = Atten_Out_4 +x
    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
x=norm(x)
FeedForward_Out_4=net(x)
x = FeedForward_Out_4 +x
#]))

#5
#layers.append(nn.ModuleList([
    #PreNorm(dim, Attention(dim, heads=heads, dim_head =dim_head, dropout=dropout)),
x=norm(x)
qkv =to_qkv(x).chunk(3, dim = -1) #(1,65,1024) *3
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv) #(1,16,65,64) *3

dots=torch.matmul(q, k.transpose(-1,-2))*scale #(1,16,65,64) (1,16,64,65)->(1,16,65,65)
attn = attend(dots) #(1,16,65,65)
out = torch.matmul(attn, v) #(1,16,65,65) (1,16,65,64)->(1,16,65,64)
out = rearrange(out, 'b h n d -> b n (h d)') #(1,65, 1024)
Atten_Out_5 =to_out(out)
x = Atten_Out_5 +x
    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
x=norm(x)
FeedForward_Out_5=net(x)
x = FeedForward_Out_5 +x
#]))

#6
#layers.append(nn.ModuleList([
    #PreNorm(dim, Attention(dim, heads=heads, dim_head =dim_head, dropout=dropout)),
x=norm(x)
qkv =to_qkv(x).chunk(3, dim = -1) #(1,65,1024) *3
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv) #(1,16,65,64) *3

dots=torch.matmul(q, k.transpose(-1,-2))*scale #(1,16,65,64) (1,16,64,65)->(1,16,65,65)
attn = attend(dots) #(1,16,65,65)
out = torch.matmul(attn, v) #(1,16,65,65) (1,16,65,64)->(1,16,65,64)
out = rearrange(out, 'b h n d -> b n (h d)') #(1,65, 1024)
Atten_Out_6 =to_out(out)
x = Atten_Out_6 +x
    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
x=norm(x)
FeedForward_Out_6=net(x)
x = FeedForward_Out_6 +x #(1,65,1024)
#])) 

################################################################################################
x=x.mean(dim = 1) if pool=='mean' else x[:, 0] #(1,1024)
x = nn.Identity()(x)
#Result=mlp_head(x)
# mlp_head = nn.Sequential(
#     nn.LayerNorm(dim),
#     nn.Linear(dim, num_classes)
# )
x=nn.LayerNorm(dim)(x)
Result=nn.Linear(dim, num_classes)(x) #(1,1000)



















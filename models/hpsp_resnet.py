import torch
import torch.nn as nn
from einops import rearrange
#from segmentation_models_pytorch.swinunet.swin_transformer import BasicLayer
#from segmentation_models_pytorch.swinunet.resnet import resnet50
from models.resnet import resnet50, resnet101
from models.swin_transformer import BasicLayer

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class Edge_PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.output_dim = dim // 16
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 8 * dim, bias=False)
        self.output_dim = dim // 2
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        # x = x.view(B,-1,self.output_dim)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchExpand_X8(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 32)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, C0, H0, W0 = x.size()
        x=x.view(B, C0, -1)
        x=x.permute(0, 2, 1)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=8, p2=8, c=C // 64)
        x = x.view(B, -1, C // 64)
        x = self.norm(x)

        return x


class PatchExpand_XX(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.scale = dim_scale
        self.norm = norm_layer(dim // (dim_scale ** 2))

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, C0, H0, W0 = x.size()
        x=x.view(B, C0, -1)
        x=x.permute(0, 2, 1)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.scale, p2=self.scale, c=C // (self.scale ** 2))
        x = x.view(B, -1, C // (self.scale ** 2))
        x = self.norm(x)

        return x


class main_model(nn.Module):
    def __init__(self, backbone, embed_dim=256, num_classes=20, num_instances=12, use_checkpoint=False, pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim
        # self.backbone=resnet50(pretrained=pretrained,if_include_top=False)
        if backbone == 'res50':
            self.backbone=resnet50(pretrained=pretrained)
        elif backbone =='res101':
            self.backbone = resnet101(pretrained=pretrained)

        del self.backbone.fc
        del self.backbone.avgpool
        # self.edge_expand=Edge_PatchExpand(dim=embed_dim)
        # self.out_edge=nn.Conv2d(in_channels=embed_dim//16,out_channels=1,kernel_size=1,bias=False)
        self.expand8 = PatchExpand_X8(dim=8 * embed_dim)
        self.expand4 = PatchExpand_XX(dim=4 * embed_dim, dim_scale=4)
        self.expand2 = PatchExpand_XX(dim=2 * embed_dim, dim_scale=2)
        self.decoder1 = BasicLayer(dim=2 * embed_dim, depth=4, num_heads=8, window_size=7, drop_path=[0.025, 0.05, 0.075, 0.1],
                                   use_checkpoint=use_checkpoint)
        self.FinalPatchExpand = FinalPatchExpand_X4(dim=2 * embed_dim)
        self.out_ins = nn.Conv2d(in_channels=embed_dim, out_channels=num_instances, kernel_size=3, padding=1)
        self.out_parsing = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=3, padding=1)
    def forward(self, x):
        B, C, H, W = x.size()
        C2,C3,C4,C5=self.backbone(x)
        #s0, s1 = self.backbone.layers[0](x, H // 4, W // 4)
        #s1_1, s2 = self.backbone.layers[1](s1, H // 8, W // 8)
        #s2_1, s3 = self.backbone.layers[2](s2, H // 16, W // 16)
        #s4 = self.backbone.layers[3](s3, H // 32, W // 32)
        e4 = self.expand8(C5, H // 32, W // 32)
        e3 = self.expand4(C4, H // 16, W // 16)
        e2 = self.expand2(C3, H // 8, W // 8)
        B, C_e2, _, _=C2.size()
        e1=C2.view(B, C_e2, -1)
        e1=e1.permute(0, 2, 1)
        #print(e4.size())
        #print(e3.size())
        #print(e2.size())
        #print(e1.size())
        e0 = torch.cat([e4, e3, e2, e1], dim=2)
        f1 = self.decoder1(e0, H // 4, W // 4)
        f2 = self.FinalPatchExpand(f1, H // 4, W // 4)
        out_ins=self.out_ins(f2)
        out_parsing = self.out_parsing(f2)

        # if torch.any(torch.isnan(out_ins)):
        #     import pdb;
        #     pdb.set_trace()

        return out_ins, out_parsing

if __name__ == "__main__":
    x = torch.randn(1, 3, 384, 384)
    model = main_model(pretrained=False)
    y3, y5 = model(x)
    print(y3.size())

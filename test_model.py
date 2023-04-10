import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import dgl
import warnings
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.to(x.device)
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.to(x.device)

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.to(x.device)

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class EdgeConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 requires_grad=True):
        assert kernel_size == 3, 'EdgeConv2d\'s kernel_size must be 3.'
        assert out_channels % 8 == 0, 'EdgeConv2d\'s out_channels must be a multiple of 8.'
        assert out_channels % groups == 0, 'EdgeConv2d\'s out_channels must be a multiple of groups'
        super().__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__dilation = dilation
        self.__groups = groups

        if requires_grad and bias:
            self.__bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)  # noqa
        else:
            self.__bias = None

        self.__weight = nn.Parameter(torch.zeros(size=(out_channels, int(
            self.__in_channels / self.__groups), self.__kernel_size, self.__kernel_size)), requires_grad=False)

        kernel_mid = self.__kernel_size // 2
        for idx in range(self.__out_channels):
            if idx % 8 == 0:
                # vertical sobel operator
                '''
                    -1  -2  -1
                    0   0   0
                    1   2   1
                '''
                self.__weight[idx, :, 0, :] = -1
                self.__weight[idx, :, 0, kernel_mid] = -2
                self.__weight[idx, :, -1, :] = 1
                self.__weight[idx, :, -1, kernel_mid] = 2
            elif idx % 8 == 1:
                # horizontal sobel operator
                '''
                    -1  0   1
                    -2  0   2
                    -1  0   1
                '''
                self.__weight[idx, :, :, 0] = -1
                self.__weight[idx, :, kernel_mid, 0] = -2
                self.__weight[idx, :, :, -1] = 1
                self.__weight[idx, :, kernel_mid, -1] = 2
            elif idx % 8 == 2:
                # diagonal sobel operator
                '''
                    -2  -1  0
                    -1  0   1
                    0   1   2
                '''
                self.__weight[idx, :, 0, 0] = -2
                self.__weight[idx, :, -1, -1] = 2
                for i in range(0, kernel_mid + 1):
                    self.__weight[idx, :, kernel_mid - i, i] = -1
                    self.__weight[idx, :, self.__kernel_size -
                                  1 - i, kernel_mid + i] = 1
            elif idx % 8 == 3:
                # anti-diagonal sobel operator
                '''
                    0   1   2
                    -1  0   1
                    -2  -1  0
                '''
                self.__weight[idx, :, -1, 0] = -2
                self.__weight[idx, :, 0, -1] = 2
                for i in range(0, kernel_mid + 1):
                    self.__weight[idx, :, i, kernel_mid + i] = 1
                    self.__weight[idx, :, kernel_mid + i, i] = -1
            elif idx % 8 == 4:
                # vertical scharr operator
                '''
                    -3  -10 -3
                    0   0   0
                    3   10  3
                '''
                self.__weight[idx, :, 0, :] = -3
                self.__weight[idx, :, 0, kernel_mid] = -10
                self.__weight[idx, :, -1, :] = 3
                self.__weight[idx, :, -1, kernel_mid] = 10
            elif idx % 8 == 5:
                # horizontal scharr operator
                '''
                    -3  0   3
                    -10 0   10
                    -3  0   3
                '''
                self.__weight[idx, :, :, 0] = -3
                self.__weight[idx, :, kernel_mid, 0] = -10
                self.__weight[idx, :, :, -1] = 3
                self.__weight[idx, :, kernel_mid, -1] = 10
            elif idx % 8 == 6:
                # laplacian operator operator1
                '''
                    0   1   0
                    1   -4  1
                    0   1   0
                '''
                self.__weight[idx, :, kernel_mid, :] = 1
                self.__weight[idx, :, :, kernel_mid] = 1
                self.__weight[idx, :, kernel_mid, kernel_mid] = -4
            else:
                # laplacian operator operator2
                '''
                    1   1   1
                    1   -8  1
                    1   1   1
                '''
                self.__weight[idx, :, :, :] = 1
                self.__weight[idx, :, kernel_mid, kernel_mid] = -8

        if requires_grad:
            self.__edge_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=True)  # noqa
        else:
            self.__edge_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=False)  # noqa

    def forward(self, x):
        self.__edge_factor = self.__edge_factor.to(x.device)
        if isinstance(self.__bias, nn.Parameter):
            self.__bias = self.__bias.to(x.device)

        weight = self.__weight * self.__edge_factor
        weight = weight.to(x.device)
        return F.conv2d(x, weight, self.__bias, self.__stride, self.__padding, self.__dilation, self.__groups)


def compute_dist(feat):
    D = torch.cdist(feat, feat, p=2)

    return D


def hard_knn(D, k):
    r"""
    input D: b m n
    output Idx: b m k
    """
    score, idx = torch.topk(D, k, dim=2, largest=False, sorted=True)
    sigma = torch.sum(score, dim=-1, keepdim=True) / k
    score = (-score / (sigma ** 2)).exp()

    return score, idx


class StaticWGAT_Layer(nn.Module):
    def __init__(self, dim):
        super(StaticWGAT_Layer, self).__init__()

        self.conv_s = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()
        )

    def message_func(self, edges):
        h = edges.src['h']
        e = edges.data['e']
        e = e.view(-1, 1, 1, 1)
        m = e * h
        m = self.conv_s(m)

        return {'m': m}

    def reduce_func(self, nodes):
        m = nodes.mailbox['m']
        h = torch.max(m, dim=1)[0]

        return {"h": h}

    def forward(self, g):
        g = g.local_var()
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata['h']


class WGAT_Layer(nn.Module):
    def __init__(self, dim):
        super(WGAT_Layer, self).__init__()
        self.static_layer = StaticWGAT_Layer(dim)

    def forward(self, g):
        # static
        out = self.static_layer(g)
        return out


class WGAT(nn.Module):
    def __init__(self, channels=128, n_layers=2, window_size=8, k=8):
        super(WGAT, self).__init__()
        self.index_neighbos_cache = {}
        self.window_size = window_size
        self.k = k
        self.layers = nn.ModuleList([WGAT_Layer(channels) for _ in range(n_layers)])

    def forward(self, x):
        x_patch = rearrange(x, 'b c (n1 w1) (n2 w2) -> b n1 n2 c w1 w2', w1=self.window_size, w2=self.window_size)
        b, n1, n2, c, _, _ = x_patch.shape

        x_nodes = rearrange(x_patch, 'b n1 n2 c w1 w2 -> b (n1 n2) c w1 w2')
        D = compute_dist(rearrange(x_nodes, 'b n c w1 w2 -> b n (c w1 w2)'))

        # hard knn
        score_k, idx_k = hard_knn(D, self.k)
        g = self.get_graphs(x_nodes, score_k, idx_k)

        for layer in self.layers:
            feat = layer(g)
            g.ndata['h'] = feat

        hg = dgl.unbatch(g)
        out = torch.cat([hg[i].ndata['h'].unsqueeze(0) for i in range(len(hg))])
        out = rearrange(out, 'b (n1 n2) c w1 w2 -> b c (n1 w1) (n2 w2)', n1=n1)

        return out

    def get_graphs(self, x_nodes, score_k, idx_k):
        graphs = []

        for i in range(len(x_nodes)):
            x_nodes_feat = x_nodes[i]

            src_ids = []
            dst_ids = []

            for dst, srcs in enumerate(idx_k[i]):
                for src in srcs:
                    src_ids.append(src.cpu())
                    dst_ids.append(dst)

            g = dgl.graph(data=(src_ids, dst_ids), num_nodes=x_nodes_feat.shape[0], device=x_nodes.device)
            g.ndata['h'] = x_nodes_feat
            g.edata['e'] = score_k[i].reshape(-1).unsqueeze(1)

            graphs.append(g)

        batched_graph = dgl.batch(graphs)

        return batched_graph


class ERA_WGAT_S(nn.Module):
    def __init__(self, config):
        super(ERA_WGAT_S, self).__init__()
        self.in_channels = 1
        self.pgat_layers = config['pgat_layers']
        self.k = config['k']

        # edge branch
        self.edge_conv = nn.Sequential(
            EdgeConv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU()
        )
        self.e_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.e_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.e_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )

        # encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

        self.edge_cat_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.edge_cat_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.edge_cat_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.edge_cat_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.enc_window_gat_128 = WGAT(channels=128, n_layers=self.pgat_layers, window_size=8, k=self.k)
        self.enc_graph_conv_128 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        # bottleneck
        self.neck_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.GELU()
        )
        self.neck_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.GELU()
        )
        self.neck_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.GELU()
        )

        self.neck_window_gat_64 = WGAT(channels=256, n_layers=self.pgat_layers, window_size=4, k=self.k)
        self.neck_graph_conv_64 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        # decoder
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dec_conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.dec_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dec_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dec_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.dec_window_gat_128 = WGAT(channels=128, n_layers=self.pgat_layers, window_size=8, k=self.k)
        self.dec_graph_conv_128 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.relu = nn.ReLU(inplace=True)

    # @autocast()
    def forward(self, x):
        e_1 = self.edge_conv(x)
        e_2 = self.e_conv1(e_1)
        e_3 = self.e_conv2(e_2)
        e_4 = self.e_conv3(e_3)

        out = self.enc_conv1(x)  # 32x512x512
        out = torch.cat([out, e_1], dim=1)  # (32+32)x512x512
        out = self.edge_cat_conv1(out)  # 32x512x512
        residual_1 = out

        out = self.enc_conv2(out)  # 64x256x256
        out = torch.cat([out, e_2], dim=1)  # (64+64)x256x256
        out = self.edge_cat_conv2(out)  # 64x256x256
        residual_2 = out

        out = self.enc_conv3(out)  # 128x128x128
        out = torch.cat([out, e_3], dim=1)  # (128+128)x128x128
        out = self.edge_cat_conv3(out)  # 128x128x128
        residual_3 = out

        # perform Patch-GAT for size 128x128 in encoder stage
        enc_hg_128 = self.enc_window_gat_128(out)
        out = torch.cat([out, enc_hg_128], dim=1)
        out = self.enc_graph_conv_128(out)

        out = self.enc_conv4(out)  # 256x64x64
        out = torch.cat([out, e_4], dim=1)  # (256+256)x64x64
        out = self.edge_cat_conv4(out)  # 256x64x64
        residual_4 = out

        out = self.neck_conv1(out)  # 256x64x64
        out = self.neck_conv2(out)  # 256x64x64

        # perform Patch-GAT for size 64x64 in neck stage
        neck_hg_64 = self.neck_window_gat_64(out)
        out = torch.cat([out, neck_hg_64], dim=1)
        out = self.neck_graph_conv_64(out)

        out = self.neck_conv3(out)  # 256x64x64

        out = torch.cat([out, residual_4], dim=1)  # (256+256)x64x64
        out = self.dec_conv1(out)  # 256x64x64

        out = self.dec_up1(out)  # 128x128x128

        out = torch.cat([out, residual_3], dim=1)  # (128+128)x128x128
        out = self.dec_conv2(out)  # 128x128x128

        # perform Patch-GAT for size 128x128 in decoder stage
        dec_hg_128 = self.dec_window_gat_128(out)
        out = torch.cat([out, dec_hg_128], dim=1)
        out = self.dec_graph_conv_128(out)

        out = self.dec_up2(out)  # 64x256x256

        out = torch.cat([out, residual_2], dim=1)  # (64+64)x256x256
        out = self.dec_conv3(out)  # 64x256x256

        out = self.dec_up3(out)  # 32x512x512

        out = torch.cat([out, residual_1], dim=1)  # (32+32)x512x512
        out = self.dec_conv4(out)  # 32x512x512

        out = self.dec_conv5(out)  # 1x512x512
        out = self.relu(out + x)

        return out




if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    config = {
        'pgat_layers': 2,
        'k': 8
    }
    model = ERA_WGAT_S(config)
    device = torch.device('cpu')
    input = torch.zeros((1, 1, 512, 512)).to(device)
    flops, params = profile(model.to(device), inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

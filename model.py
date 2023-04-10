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


# class StaticWGAT_Layer(nn.Module):
#     def __init__(self, dim):
#         super(StaticWGAT_Layer, self).__init__()
#
#         self.conv_s = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.GELU()
#         )
#
#     def message_func(self, edges):
#         h = edges.src['h']
#         e = edges.data['e']
#         e = e.view(-1, 1, 1, 1)
#         m = e * h
#         m = self.conv_s(m)
#
#         return {'m': m}
#
#     def reduce_func(self, nodes):
#         m = nodes.mailbox['m']
#         # h = torch.mean(m, dim=1)
#         h = torch.max(m, dim=1)[0]
#         # h = F.layer_norm(h, h.size()[1:])
#
#         return {"h": h}
#
#     def forward(self, g):
#         g = g.local_var()
#         g.update_all(self.message_func, self.reduce_func)
#
#         return g.ndata['h']
#
#
# class DynamicWGAT_Layer(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(DynamicWGAT_Layer, self).__init__()
#
#         # dynamic attention
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, kernel_size=1),
#             nn.GELU()
#         )
#         self.atten_conv = nn.Sequential(
#             nn.Conv2d(out_dim * 2, 1, kernel_size=1),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#
#     def edge_attention(self, edges):
#         z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
#         attention = self.atten_conv(z2)
#         attention = F.leaky_relu(attention)
#         attention = torch.flatten(attention, 1)
#
#         return {'a': attention}
#
#     def message_func(self, edges):
#         return {'z': edges.src['z'], 'a': edges.data['a']}
#
#     def reduce_func(self, nodes):
#         alpha = F.softmax(nodes.mailbox['a'], dim=1)
#         b, k, _ = alpha.shape
#         alpha = alpha.view(b, k, 1, 1, 1)
#         # h = torch.max(alpha * nodes.mailbox['z'], dim=1)[0]
#         h = torch.mean(alpha * nodes.mailbox['z'], dim=1)
#         # h = F.layer_norm(h, h.size()[1:])
#
#         return {'h': h}
#
#     def forward(self, graph):
#         graph = graph.local_var()
#         feat = graph.ndata['h']
#         graph.ndata['z'] = self.conv_1(feat)
#         graph.apply_edges(self.edge_attention)
#         graph.update_all(self.message_func, self.reduce_func)
#
#         return graph.ndata['h']


# class StaticWGAT_Layer(nn.Module):
#     def __init__(self, dim):
#         super(StaticWGAT_Layer, self).__init__()
#
#         self.conv_s = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#             nn.GELU()
#         )
#
#     def message_func(self, edges):
#         h = edges.src['h']
#         e = edges.data['e']
#         e = e.view(-1, 1, 1, 1)
#         m = e * h
#         m = self.conv_s(m)
#
#         return {'m': m}
#
#     def reduce_func(self, nodes):
#         m = nodes.mailbox['m']
#         h = torch.max(m, dim=1)[0]
#
#         return {"h": h}
#
#     def forward(self, g):
#         g = g.local_var()
#         g.update_all(self.message_func, self.reduce_func)
#
#         return g.ndata['h']
#
#
# class DynamicWGAT_Layer(nn.Module):
#     def __init__(self, dim):
#         super(DynamicWGAT_Layer, self).__init__()
#
#         # dynamic attention
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#             nn.GELU()
#         )
#
#         self.atten_conv = nn.Sequential(
#             nn.Conv2d(dim * 2, 1, kernel_size=3, padding=1),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#
#     def edge_attention(self, edges):
#         h = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
#         attention = self.atten_conv(h)
#         attention = F.leaky_relu(attention)
#         attention = torch.flatten(attention, 1)
#
#         return {'a': attention}
#
#     def message_func(self, edges):
#         return {'h': edges.src['h'], 'a': edges.data['a']}
#
#     def reduce_func(self, nodes):
#         alpha = F.softmax(nodes.mailbox['a'], dim=1)
#         b, k, _ = alpha.shape
#         alpha = alpha.view(b, k, 1, 1, 1)
#         h = alpha * nodes.mailbox['h']
#         h = rearrange(h, 'b k c h w -> (b k) c h w')
#         h = self.conv(h)
#         h = rearrange(h, '(b k) c h w -> b k c h w', b=b)
#         h = torch.mean(h, dim=1)
#         # h = torch.max(h, dim=1)[0]
#
#         return {'h': h}
#
#     def forward(self, graph):
#         graph = graph.local_var()
#         graph.apply_edges(self.edge_attention)
#         graph.update_all(self.message_func, self.reduce_func)
#
#         return graph.ndata['h']


class WGAT_Layer(nn.Module):
    def __init__(self, dim):
        super(WGAT_Layer, self).__init__()
        self.conv_s = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv_d = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

    def message_func(self, edges):
        s = edges.src['s']
        e = edges.data['e']
        e = e.view(-1, 1, 1, 1)
        m = e * s
        m = self.conv_s(m)

        return {'d': edges.src['d'], 'm': m, 'a': edges.data['a']}

    def reduce_func(self, nodes):
        m = nodes.mailbox['m']
        s = torch.max(m, dim=1)[0]

        alpha = F.softmax(nodes.mailbox['a'], dim=1)
        b, k, _ = alpha.shape
        alpha = alpha.view(b, k, 1, 1, 1)
        d = torch.sum(alpha * nodes.mailbox['d'], dim=1)

        return {'s': s, 'd': d}

    def edge_attention(self, edges):
        z = torch.cat([edges.src['d'], edges.dst['d']], dim=1)
        attention = self.conv_atten(z)
        attention = F.leaky_relu(attention)
        attention = torch.flatten(attention, 1)

        return {'a': attention}

    def forward(self, g):
        g = g.local_var()
        d = g.ndata['d']
        d = self.conv_d(d)
        g.ndata['d'] = d
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata['s'], g.ndata['d']


# class SFF(nn.Module):
#     def __init__(self,
#                  in_channel,
#                  reduction=2):
#         super().__init__()
#         self.__fc = nn.Linear(in_channel, in_channel // reduction)
#         self.__fc_x = nn.Linear(in_channel // reduction, in_channel)
#         self.__fc_y = nn.Linear(in_channel // reduction, in_channel)
#         self.__softmax = nn.Softmax(dim=1)
#
#     def forward(self, x, y):
#         u = x + y
#         u = u.mean(-1).mean(-1)
#         u = self.__fc(u)
#         attentionVec = torch.cat([self.__fc_x(u).unsqueeze_(1), self.__fc_y(u).unsqueeze(1)], dim=1)  # noqa
#         attentionVec = self.__softmax(attentionVec).unsqueeze(-1).unsqueeze(-1)
#         out = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
#         out = (out * attentionVec).sum(dim=1)
#         return out


class SFF(nn.Module):
    def __init__(self,
                 in_channel,
                 reduction=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=3),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1),
            nn.GELU()
        )
        self.conv2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x, y):
        u = torch.cat([x, y], dim=1)
        u = self.conv1(u)
        attentionVec = torch.cat([self.conv2(u).unsqueeze_(1), self.conv3(u).unsqueeze(1)], dim=1)
        attentionVec = self.softmax(attentionVec)
        out = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        out = out * attentionVec
        out = rearrange(out, 'b m c h w -> b (m c) h w')
        out = self.conv4(out)
        return out


class CAF(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(CAF, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        x_avg = self.avgpool(x).squeeze(-1).squeeze(-1)
        x_max = self.maxpool(x).squeeze(-1).squeeze(-1)
        x_attention = self.fc(x_avg) + self.fc(x_max)
        x_attention = F.sigmoid(x_attention)
        x_attention = x_attention.unsqueeze(-1).unsqueeze(-1)
        out = x * x_attention.expand_as(x)
        return out


# class WGAT_Layer(nn.Module):
#     def __init__(self, dim):
#         super(WGAT_Layer, self).__init__()
#         self.static_layer = StaticWGAT_Layer(dim)
#         # # self.dynamic_layer_heads = nn.ModuleList([DynamicWGAT_Layer(dim, dim // num_heads) for _ in range(num_heads)])
#         self.dynamic_layer = DynamicWGAT_Layer(dim)
#         # self.caf = CAF(dim)
#         # self.sff = SFF(dim)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
#             nn.GELU()
#         )
#         # self.conv = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
#         # self.gelu = nn.GELU()
#         # self.layers = SD_WGAT_Layer(dim, dim)
#
#     def forward(self, g):
#         # static
#         feat_static = self.static_layer(g)
#         # dynamic
#         feat_dynamic = self.dynamic_layer(g)
#         # out = self.sff(feat_static, feat_dynamic)
#         # out = torch.cat([feat_static, feat_dynamic], dim=1)
#         # out = self.conv(out)
#         # out = self.gelu(out)
#         out = torch.cat([feat_static, feat_dynamic], dim=1)
#         out = self.conv(out)
#         # out = self.caf(feat_static) + self.caf(feat_dynamic)
#
#         return out


class WGAT(nn.Module):
    def __init__(self, channels=128, n_layers=2, window_size=8, k=8):
        super(WGAT, self).__init__()
        self.index_neighbos_cache = {}
        self.window_size = window_size
        self.k = k
        self.layers = nn.ModuleList([WGAT_Layer(channels) for _ in range(n_layers)])
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        x_patch = rearrange(x, 'b c (n1 w1) (n2 w2) -> b n1 n2 c w1 w2', w1=self.window_size, w2=self.window_size)
        b, n1, n2, c, _, _ = x_patch.shape

        x_nodes = rearrange(x_patch, 'b n1 n2 c w1 w2 -> b (n1 n2) c w1 w2')
        D = compute_dist(rearrange(x_nodes, 'b n c w1 w2 -> b n (c w1 w2)'))

        # hard knn
        score_k, idx_k = hard_knn(D, self.k)
        g = self.get_graphs(x_nodes, score_k, idx_k)

        for layer in self.layers:
            feat_static, feat_dynamic = layer(g)
            g.ndata['s'] = feat_static
            g.ndata['d'] = feat_dynamic

        hg = dgl.unbatch(g)
        s = torch.cat([hg[i].ndata['s'].unsqueeze(0) for i in range(len(hg))])
        s = rearrange(s, 'b (n1 n2) c w1 w2 -> b c (n1 w1) (n2 w2)', n1=n1)
        d = torch.cat([hg[i].ndata['d'].unsqueeze(0) for i in range(len(hg))])
        d = rearrange(d, 'b (n1 n2) c w1 w2 -> b c (n1 w1) (n2 w2)', n1=n1)

        out = torch.cat([s, d], dim=1)
        out = self.conv(out)
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
            g.ndata['s'] = x_nodes_feat
            g.ndata['d'] = x_nodes_feat
            g.edata['e'] = score_k[i].reshape(-1).unsqueeze(1)

            graphs.append(g)

        batched_graph = dgl.batch(graphs)

        return batched_graph


class ERA_WGAT(nn.Module):
    def __init__(self, config):
        super(ERA_WGAT, self).__init__()
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

    @autocast()
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
        'pgat_layers': 6,
        'k': 8
    }
    model = ERA_WGAT(config)
    device = torch.device('cpu')
    input = torch.zeros((1, 1, 512, 512)).to(device)
    flops, params = profile(model.to(device), inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

# --------------------------------------------------------
# GMD-YOLO 专属极致边缘重参数化主干 (RepViT-Edge)
# 针对地下管道机器人 (NPU/嵌入式工控机) 进行了底层硬件级重构
# 1. 彻底剔除 GELU，采用 ReLU/Hardswish
# 2. 移除 timm 依赖，重写 LiteSE 模块，采用极速 Hardsigmoid 替换 Sigmoid
# 3. 针对中分辨率层进行深度非对称剪枝
# --------------------------------------------------------

import torch
import torch.nn as nn

__all__ = ['repvit_edge_nano', 'repvit_edge_micro']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ==========================================================
# 【终极优化 1】：自研极速通道注意力 (LiteSE)
# 抛弃 timm 库，消灭 Sigmoid 中的 e^-x 指数运算，全面拥抱 Hardsigmoid
# ==========================================================
class LiteSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = _make_divisible(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, channels, 1, 1, 0, bias=True)
        # 边缘端神器：用极其廉价的分段线性函数替代昂贵的 Sigmoid
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.hsigmoid(out)
        return x * out


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse_self(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:],
                            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse_self(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse_self()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse_self(self):
        conv = self.conv.fuse_self()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        # 【核心优化 2】：保留硬件亲和的 ReLU 和 Hardswish
        activation_layer = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                LiteSE(inp, 4) if use_se else nn.Identity(),  # 使用自研 LiteSE
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                activation_layer,
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                LiteSE(inp, 4) if use_se else nn.Identity(),  # 使用自研 LiteSE
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                activation_layer,
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViT_Edge(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        input_channel = self.cfgs[0][2]

        # 极简前馈起点，纯 ReLU
        patch_embed = torch.nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
        )
        layers = [patch_embed]

        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(RepViTBlock(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        # 精准获取 4 个 Stage 的特征维度供 YOLO 使用
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        results = [None, None, None, None]
        temp = None
        i = None
        for index, f in enumerate(self.features):
            x = f(x)
            if index == 0:
                temp = x.size(1)
                i = 0
            elif x.size(1) == temp:
                results[i] = x
            else:
                temp = x.size(1)
                i = i + 1
        return results


# =========================================================================
# 实例化接口区 (直接在 YOLO 的 yaml 中调用以下两个名字之一)
# =========================================================================

def repvit_edge_micro():
    """
    【均衡优选版】
    基础通道 [40, 80, 160, 320]。
    对 Stage 3 进行了深度剪枝，参数预估 2.0M~2.5M，极其适合携带 MADR。
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 40, 1, 0, 1], [3, 2, 40, 0, 0, 1],  # P2
        [3, 2, 80, 0, 0, 2], [3, 2, 80, 1, 0, 1], [3, 2, 80, 0, 0, 1],  # P3
        [3, 2, 160, 0, 1, 2], [3, 2, 160, 1, 1, 1], [3, 2, 160, 0, 1, 1], [3, 2, 160, 1, 1, 1],  # P4 (剪枝至 4 层)
        [3, 2, 320, 0, 1, 2], [3, 2, 320, 1, 1, 1],  # P5
    ]
    return RepViT_Edge(cfgs)


def repvit_edge_nano():
    """
    【极限缩胸版】
    基础通道 [32, 64, 128, 256]。
    当算力面临绝境时的保底方案，参数预估 1.0M~1.5M，极致榨干最后一滴性能。
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 32, 1, 0, 1], [3, 2, 32, 0, 0, 1],
        [3, 2, 64, 0, 0, 2], [3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 1, 2], [3, 2, 128, 1, 1, 1], [3, 2, 128, 0, 1, 1], [3, 2, 128, 1, 1, 1],
        [3, 2, 256, 0, 1, 2], [3, 2, 256, 1, 1, 1],
    ]
    return RepViT_Edge(cfgs)


# --------------------------------------------------------
# GMD-YOLO 专属极致边缘重参数化主干 (RepViT-Edge)
# 针对地下管道机器人 (NPU/嵌入式工控机) 进行了底层硬件级重构
# 1. 彻底剔除 GELU，采用 ReLU/Hardswish
# 2. 移除 timm 依赖，重写 LiteSE 模块，采用极速 Hardsigmoid 替换 Sigmoid
# 3. 针对中分辨率层进行深度非对称剪枝
# 4. [已修复] 完整的结构重参数化逻辑，支持一键切换部署模式
# --------------------------------------------------------

# import torch
# import torch.nn as nn
#
# __all__ = ['repvit_edge_nano', 'repvit_edge_micro']
#
#
# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# # ==========================================================
# # 【终极优化 1】：自研极速通道注意力 (LiteSE)
# # ==========================================================
# class LiteSE(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         mid_channels = _make_divisible(channels // reduction, 8)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(channels, mid_channels, 1, 1, 0, bias=True)
#         self.act = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(mid_channels, channels, 1, 1, 0, bias=True)
#         self.hsigmoid = nn.Hardsigmoid(inplace=True)
#
#     def forward(self, x):
#         out = self.pool(x)
#         out = self.conv1(out)
#         out = self.act(out)
#         out = self.conv2(out)
#         out = self.hsigmoid(out)
#         return x * out
#
#
# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
#                  groups=1, bn_weight_init=1):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         self.add_module('bn', torch.nn.BatchNorm2d(b))
#         torch.nn.init.constant_(self.bn.weight, bn_weight_init)
#         torch.nn.init.constant_(self.bn.bias, 0)
#         self.deploy = False  # 新增部署标志
#
#     @torch.no_grad()
#     def fuse_self(self):
#         c, bn = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps) ** 0.5
#         w = c.weight * w[:, None, None, None]
#         b = bn.bias - bn.running_mean * bn.weight / \
#             (bn.running_var + bn.eps) ** 0.5
#         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:],
#                             stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
#                             groups=self.c.groups, device=c.weight.device)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
#
#     @torch.no_grad()
#     def switch_to_deploy(self):
#         if self.deploy:
#             return
#         # 融合卷积和BN
#         fused_conv = self.fuse_self()
#         self.c = fused_conv
#         self.__delattr__('bn')  # 删除BN层
#         self.deploy = True
#
#     def forward(self, x):
#         if self.deploy:
#             return self.c(x)
#         return super().forward(x)
#
# class Residual(torch.nn.Module):
#     def __init__(self, m, drop=0.):
#         super().__init__()
#         self.m = m
#         self.drop = drop
#
#     def forward(self, x):
#         if self.training and self.drop > 0:
#             return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
#                                               device=x.device).ge_(self.drop).div(1 - self.drop).detach()
#         else:
#             return x + self.m(x)
#
#     #
#     # @torch.no_grad()
#     # def fuse_self(self):
#     #     if isinstance(self.m, Conv2d_BN):
#     #         m = self.m.fuse_self()
#     #         assert (m.groups == m.in_channels)
#     #         identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
#     #         identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
#     #         m.weight += identity.to(m.weight.device)
#     #         return m
#     #     elif isinstance(self.m, torch.nn.Conv2d):
#     #         m = self.m
#     #         assert (m.groups != m.in_channels)
#     #         identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
#     #         identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
#     #         m.weight += identity.to(m.weight.device)
#     #         return m
#     #     else:
#     #         return self
#     #
#     # @torch.no_grad()
#     # def switch_to_deploy(self):
#     #     if self.deploy:
#     #         return
#     #     # 获取融合了 identity 的卷积
#     #     fused_m = self.fuse_self()
#     #     self.m = fused_m
#     #     self.deploy = True
#
#
# class RepVGGDW(torch.nn.Module):
#     def __init__(self, ed) -> None:
#         super().__init__()
#         self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
#         self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
#         self.bn = torch.nn.BatchNorm2d(ed)
#         self.deploy = False
#
#     def forward(self, x):
#         if self.deploy:
#             return self.rbr_reparam(x)
#         return self.bn((self.conv(x) + self.conv1(x)) + x)
#
#     @torch.no_grad()
#     def fuse_self(self):
#         conv = self.conv.fuse_self()
#         conv1 = self.conv1
#
#         conv_w = conv.weight
#         conv_b = conv.bias
#         conv1_w = conv1.weight
#         conv1_b = conv1.bias
#
#         conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
#         identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
#                                            [1, 1, 1, 1])
#
#         final_conv_w = conv_w + conv1_w + identity
#         final_conv_b = conv_b + conv1_b
#
#         conv.weight.data.copy_(final_conv_w)
#         conv.bias.data.copy_(final_conv_b)
#
#         bn = self.bn
#         w = bn.weight / (bn.running_var + bn.eps) ** 0.5
#         w = conv.weight * w[:, None, None, None]
#         b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
#             (bn.running_var + bn.eps) ** 0.5
#         conv.weight.data.copy_(w)
#         conv.bias.data.copy_(b)
#         return conv
#
#     @torch.no_grad()
#     def switch_to_deploy(self):
#         if self.deploy:
#             return
#         # 调用原有的 fuse_self 获取彻底融合后的单路卷积
#         fused_conv = self.fuse_self()
#         self.rbr_reparam = fused_conv
#
#         # 删除训练时的多分支，彻底释放内存
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__('conv')
#         self.__delattr__('conv1')
#         self.__delattr__('bn')
#
#         self.deploy = True
#
#
# class RepViTBlock(nn.Module):
#     def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
#         super(RepViTBlock, self).__init__()
#         assert stride in [1, 2]
#         self.identity = stride == 1 and inp == oup
#
#         activation_layer = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)
#
#         if stride == 2:
#             self.token_mixer = nn.Sequential(
#                 Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
#                 LiteSE(inp, 4) if use_se else nn.Identity(),
#                 Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
#             )
#             self.channel_mixer = Residual(nn.Sequential(
#                 Conv2d_BN(oup, 2 * oup, 1, 1, 0),
#                 activation_layer,
#                 Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
#             ))
#         else:
#             assert (self.identity)
#             self.token_mixer = nn.Sequential(
#                 RepVGGDW(inp),
#                 LiteSE(inp, 4) if use_se else nn.Identity(),
#             )
#             self.channel_mixer = Residual(nn.Sequential(
#                 Conv2d_BN(inp, hidden_dim, 1, 1, 0),
#                 activation_layer,
#                 Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
#             ))
#
#     def forward(self, x):
#         return self.channel_mixer(self.token_mixer(x))
#
#
# class RepViT_Edge(nn.Module):
#     def __init__(self, cfgs):
#         super().__init__()
#         self.cfgs = cfgs
#         input_channel = self.cfgs[0][2]
#
#         patch_embed = torch.nn.Sequential(
#             Conv2d_BN(3, input_channel // 2, 3, 2, 1),
#             nn.ReLU(inplace=True),
#             Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
#         )
#         layers = [patch_embed]
#
#         for k, t, c, use_se, use_hs, s in self.cfgs:
#             output_channel = _make_divisible(c, 8)
#             exp_size = _make_divisible(input_channel * t, 8)
#             layers.append(RepViTBlock(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
#             input_channel = output_channel
#
#         self.features = nn.ModuleList(layers)
#         self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
#
#     def forward(self, x):
#         results = [None, None, None, None]
#         temp = None
#         i = None
#         for index, f in enumerate(self.features):
#             x = f(x)
#             if index == 0:
#                 temp = x.size(1)
#                 i = 0
#             elif x.size(1) == temp:
#                 results[i] = x
#             else:
#                 temp = x.size(1)
#                 i = i + 1
#         return results
#
#     # ==========================================================
#     # 【核心新增】：一键遍历整个网络，执行结构重参数化
#     # ==========================================================
#     @torch.no_grad()
#     def switch_to_deploy(self):
#         for m in self.modules():
#             # 核心修复：增加 m is not self 判断，防止无限递归
#             if hasattr(m, 'switch_to_deploy') and m is not self:
#                 m.switch_to_deploy()
#         print("RepViT-Edge: 所有多分支结构与BN层已成功折叠为单路卷积部署模式！")
#
#
# # =========================================================================
# # 实例化接口区
# # =========================================================================
#
# def repvit_edge_micro():
#     cfgs = [
#         [3, 2, 40, 1, 0, 1], [3, 2, 40, 0, 0, 1],
#         [3, 2, 80, 0, 0, 2], [3, 2, 80, 1, 0, 1], [3, 2, 80, 0, 0, 1],
#         [3, 2, 160, 0, 1, 2], [3, 2, 160, 1, 1, 1], [3, 2, 160, 0, 1, 1], [3, 2, 160, 1, 1, 1],
#         [3, 2, 320, 0, 1, 2], [3, 2, 320, 1, 1, 1],
#     ]
#     return RepViT_Edge(cfgs)
#
#
# def repvit_edge_nano():
#     cfgs = [
#         [3, 2, 32, 1, 0, 1], [3, 2, 32, 0, 0, 1],
#         [3, 2, 64, 0, 0, 2], [3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1],
#         [3, 2, 128, 0, 1, 2], [3, 2, 128, 1, 1, 1], [3, 2, 128, 0, 1, 1], [3, 2, 128, 1, 1, 1],
#         [3, 2, 256, 0, 1, 2], [3, 2, 256, 1, 1, 1],
#     ]
#     return RepViT_Edge(cfgs)
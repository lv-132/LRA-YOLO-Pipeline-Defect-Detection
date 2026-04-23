# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .RFAConv import *
# from .DSConv import *
# from ultralytics.nn.modules.conv import *
#
#
#
# __all__ = (['MADR_Bottleneck', 'C2f_MADR'])  # 暴露给外部调用
#
#
# class MADR_Bottleneck(nn.Module):
#
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # 隐藏层通道数
#         self.cv1 = Conv(c1, c_, k[0], 1)
#
#         # === 专家 1：线状/拓扑形态专家 (处理裂缝、树根) ===
#         # 使用 DySnakeConv 提取形变特征，并通过 1x1 卷积将通道数对齐回 c2
#         self.expert_linear = nn.Sequential(
#             DySnakeConv(c_, c2, k[1]),
#             Conv(c2 * 3, c2, k=1)
#         )
#
#         # === 专家 2：块状/结构形态专家 (处理错口、障碍物) ===
#         # 使用 RFAConv 提取高频边界与纹理特征
#         self.expert_blob = RFAConv(c_, c2, kernel_size=k[1])
#
#         # === 形态感知路由中心 (Gating Network) ===
#         mid_channels = max(c2 // 4, 16)  # 路由网络的中间降维通道
#         self.gap = nn.AdaptiveAvgPool2d(1)  # 全局池化，提取整张图的形态上下文
#         self.fc1 = nn.Conv2d(c2, mid_channels, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(mid_channels, 2, 1, bias=False)  # 输出 2 个专家的权重
#
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#
#         # 1. 两位专家并行提取特征
#         out_linear = self.expert_linear(x1)
#         out_blob = self.expert_blob(x1)
#
#         # 2. 软路由评分机制
#         U = out_linear + out_blob  # 特征相加，融合信息
#         S = self.gap(U)  # 压缩为形态向量
#         weight = self.fc2(self.relu(self.fc1(S)))  # 计算权重 [B, 2, 1, 1]
#         weight = F.softmax(weight, dim=1)  # 归一化，确保两者权重之和为 1
#
#         # 3. 特征动态重标定
#         w_linear = weight[:, 0:1, :, :]
#         w_blob = weight[:, 1:2, :, :]
#         out_fused = w_linear * out_linear + w_blob * out_blob
#
#         # 4. 残差连接
#         return x + out_fused if self.add else out_fused
#
#
# class C2f_MADR(nn.Module):
#     """
#     加入了 MADR 模块的 C2f 结构，可直接用于 YOLOv8 的 YAML 配置文件
#     """
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)
#         # 核心改动：将原本的 Bottleneck 替换为我们的 MADR_Bottleneck
#         self.m = nn.ModuleList(MADR_Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .RFAConv import *
from .DSConv import *
from ultralytics.nn.modules.conv import *

__all__ = (['MADR_Bottleneck', 'C2f_MADR'])  # 暴露给外部调用


# =========================================================================
# 【核心创新】: 多光谱频域池化 (Multi-Spectral Pooling) 所需的 DCT 基底生成函数
# =========================================================================
def build_1d_dct(i, freq, L):
    """计算 1D DCT 权重"""
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(channel, f_indices, base_size=64):
    """
    预计算 2D-DCT (二维离散余弦变换) 基底权重矩阵。
    为了适应 YOLOv8 的多尺度特征图 (如 P3, P4, P5 分辨率不同)，
    我们预计算一个基础尺寸 (如 64x64) 的基底，在 forward 时再通过双线性插值动态对齐。
    """
    dct_weights = torch.zeros(1, channel, base_size, base_size)

    # 将通道等分为 N 组，每组负责提取一种特定的频率特征
    num_freq = len(f_indices)
    c_part = channel // num_freq

    for i, (u_x, v_y) in enumerate(f_indices):
        for t_x in range(base_size):
            for t_y in range(base_size):
                # 依据 2D-DCT 公式计算当前坐标的能量权重
                weight = build_1d_dct(t_x, u_x, base_size) * build_1d_dct(t_y, v_y, base_size)

                # 处理通道数不能被整除的情况，最后一组包含所有剩余通道
                start_c = i * c_part
                end_c = (i + 1) * c_part if i < num_freq - 1 else channel
                dct_weights[:, start_c:end_c, t_x, t_y] = weight

    return dct_weights


class MultiSpectralPooling(nn.Module):
    """
    多光谱频域池化层 (替代传统的 Global Average Pooling)
    用于为后续的路由网络提供包含高频(裂缝)和低频(沉积物)的综合特征描述子。
    """

    def __init__(self, channel, base_size=64):
        super().__init__()
        # 选取 3 个关键频率进行空间信息解耦:
        # [0, 0]: 直流分量/低频，表征整体光照和大块/平缓的错口沉积物
        # [0, 1]: 水平高频，表征横向锐利边缘（横向裂缝）
        # [1, 0]: 垂直高频，表征纵向锐利边缘（纵向裂缝）
        self.f_indices = [[0, 0], [0, 1], [1, 0]]

        # 使用 register_buffer 确保权重随模型保存，并在正确的 device 上，但无需梯度更新(截断梯度)
        self.register_buffer('dct_weight', get_dct_weights(channel, self.f_indices, base_size))

    def forward(self, x):
        b, c, h, w = x.shape

        # 动态插值，将预计算的 DCT 基底对齐到当前特征图的实际分辨率 (支持多尺度训练和推理)
        weight = F.interpolate(self.dct_weight, size=(h, w), mode='bilinear', align_corners=False)

        # 频域特征提取：哈达玛乘积 (Element-wise) 后在空间维度 (H, W) 求和
        # 这一步彻底取代了 GAP 的求平均操作，保留了极度关键的高频突变信息
        out = torch.sum(x * weight, dim=[2, 3])  # 输出 shape: [B, C]

        # 恢复为全连接层需要的空间维度形状 [B, C, 1, 1]
        return out.view(b, c, 1, 1)


# =========================================================================
# 【重构的路由核心】: 频域感知动态路由瓶颈层 (Frequency-Guided MADR Bottleneck)
# =========================================================================
class MADR_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, k[0], 1)

        # === 专家 1：线状/拓扑形态专家 (处理高频特征：如裂缝、树根) ===
        # 使用 DySnakeConv 提取形变特征，并通过 1x1 卷积将通道数对齐回 c2
        self.expert_linear = nn.Sequential(
            DySnakeConv(c_, c2, k[1]),
            Conv(c2 * 3, c2, k=1)
        )

        # === 专家 2：块状/结构形态专家 (处理低/中频特征：如错口、障碍物) ===
        # 使用 RFAConv 提取边界与纹理特征
        self.expert_blob = RFAConv(c_, c2, kernel_size=k[1])

        # === 【核心升级】：频域感知路由中心 (Frequency-Guided Gating Network) ===
        mid_channels = max(c2 // 4, 16)  # 路由网络的中间降维通道

        # 废弃原有的 self.gap = nn.AdaptiveAvgPool2d(1)
        # 引入刚定义的频域池化模块，通道数与融合前的特征通道 c2 保持一致
        self.freq_pool = MultiSpectralPooling(channel=c2)

        self.fc1 = nn.Conv2d(c2, mid_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, 2, 1, bias=False)  # 输出 2 个专家的权重

        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 = self.cv1(x)

        # 1. 两位专家并行提取特征
        out_linear = self.expert_linear(x1)
        out_blob = self.expert_blob(x1)

        # 2. 频域引导的软路由评分机制
        U = out_linear + out_blob  # 预融合特征 [B, c2, H, W]

        # 通过 2D-DCT 基底提取多光谱描述子 (包含了 GAP 丢失的裂缝高频信息)
        S_freq = self.freq_pool(U)  # 输出形状 [B, c2, 1, 1]

        # 计算权重 [B, 2, 1, 1]
        weight = self.fc2(self.relu(self.fc1(S_freq)))
        weight = F.softmax(weight, dim=1)  # 归一化，确保两者权重之和为 1

        # 3. 特征动态重标定
        w_linear = weight[:, 0:1, :, :]
        w_blob = weight[:, 1:2, :, :]
        out_fused = w_linear * out_linear + w_blob * out_blob

        # 4. 残差连接
        return x + out_fused if self.add else out_fused


# =========================================================================
# 兼容 YOLOv8 yaml 调用的外层封装
# =========================================================================
class C2f_MADR(nn.Module):
    """
    加入了 频域感知 MADR 模块的 C2f 结构，可直接用于 YOLOv8 的 YAML 配置文件
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 内部循环调用我们重构好的频域 MADR_Bottleneck
        self.m = nn.ModuleList(MADR_Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
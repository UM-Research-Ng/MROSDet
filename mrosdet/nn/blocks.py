# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# 为 MROSDet 抽取基础层，2026-09-29；保留原计算逻辑。
"""MROSDet 所需的基础卷积、特征聚合与注意力层。"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # 卷积核、填充、膨胀率
    """按膨胀后的有效卷积核计算默认填充；输出尺寸仍取决于步长。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 有效卷积核尺寸
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 默认对称填充
    return p

class Conv(nn.Module):
    """卷积、批归一化与激活组合；默认使用 SiLU。"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """设置输入/输出通道、卷积核、步长、分组、膨胀率和激活方式。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """依次执行卷积、批归一化和激活。"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """跳过批归一化；仅在外部已正确融合卷积与归一化参数后使用。"""
        return self.act(self.conv(x))

class DWConv(Conv):
    """以输入/输出通道数的最大公约数分组；通道数相同时为逐通道卷积。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """按通道数确定分组，其余参数沿用 Conv。"""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class Index(nn.Module):
    """从多分支输出中选择指定位置的元素。"""

    def __init__(self, index=0):
        """记录要选择的元素索引。"""
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """返回输入序列中指定索引的张量。"""
        return x[self.index]

class Bottleneck(nn.Module):
    """两层卷积瓶颈；仅在启用 shortcut 且通道数相同时加入残差。"""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """用 e 确定隐藏通道数，k 分别指定两层卷积核。"""
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行瓶颈变换，并按配置叠加输入残差。"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """将投影特征分为两路，逐级收集瓶颈输出后拼接融合。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """建立 n 个瓶颈，隐藏通道数为 int(c2 * e)。"""
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """拼接两路初始特征及各瓶颈输出，再投影到目标通道数。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """与 forward 使用相同路径，但以显式 split 替代 chunk。"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """一条瓶颈主路与一条直接投影支路，拼接后用卷积融合。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """建立两条投影支路和 n 个瓶颈。"""
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """融合瓶颈主路与直接投影支路的特征。"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """在 C3 结构中使用指定卷积核大小的瓶颈。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """将内部瓶颈替换为两层均使用 k 卷积核的结构。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    """沿用 C2f 聚合结构，内部可选普通瓶颈、C3k 或带注意力的瓶颈。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """按 c3k 和 attn 选择内部模块；attn 为真时优先使用注意力路径。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

# SPPF 来源：Glenn Jocher 的 YOLOv5 实现。
class SPPF(nn.Module):
    """快速空间金字塔池化，通过串联池化聚合多尺度上下文。"""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """连续执行 n 次最大池化并拼接各级特征。

        默认 k=5、n=3 时，池化的有效感受野对应 5、9、13；
        仅在 shortcut 启用且输入/输出通道相同时加入残差。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """融合初始投影与连续池化结果，再按配置加入输入残差。"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y

class Attention(nn.Module):
    """在全部 H×W 位置上计算多头自注意力，并加入逐通道卷积位置编码。"""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """按 num_heads 划分通道，以 attn_ratio 确定每个头的查询/键维度。"""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算空间注意力，叠加值特征的位置编码并投影回原通道数。"""
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    """位置敏感注意力块：依次执行多头注意力和逐点前馈网络。"""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """构建注意力与前馈网络，shortcut 控制两个阶段的残差连接。"""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次执行注意力和前馈变换，并按配置使用残差连接。"""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    """将投影特征分为两路，只对其中一路堆叠 PSABlock，再拼接融合。"""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """要求输入/输出通道相同，以 e 确定分支宽度并堆叠 n 个注意力块。"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """保留一路特征，另一路经注意力处理后拼接投影。"""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class DFL(nn.Module):
    """将离散距离分布转为期望值的 DFL 积分层。

    算法出处：Generalized Focal Loss，https://ieeexplore.ieee.org/document/9792391 。
    当前 MROSDet 配置 reg_max=1，检测头使用 Identity，不经过本层。
    """

    def __init__(self, c1: int = 16):
        """用固定的 0 到 c1-1 权重计算分布期望，权重不参与训练。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, 4*c1, A) 的距离 logits 转为 (B, 4, A) 的四边距离。"""
        b, _, a = x.shape  # 批量、通道、网格点
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

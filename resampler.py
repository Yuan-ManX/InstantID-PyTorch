import math
import torch
import torch.nn as nn


# ===========================
# 前馈神经网络（Feed-Forward Network, FFN）
# ===========================

def FeedForward(dim, mult=4):
    """
    构建一个前馈神经网络（FFN）层，通常用于Transformer模型中。

    Args:
        dim (int): 输入和输出的维度大小。
        mult (int, optional): 内部维度相对于输入维度的倍数，默认为4。

    Returns:
        nn.Sequential: 包含以下层的前馈神经网络：
            - LayerNorm: 对输入进行层归一化。
            - Linear: 线性变换，将维度从dim扩展到inner_dim。
            - GELU: 高斯误差线性单元激活函数。
            - Linear: 线性变换，将维度从inner_dim恢复到dim。
    """
    # 计算内部维度，通常是输入维度的4倍
    inner_dim = int(dim * mult)

    return nn.Sequential(
        nn.LayerNorm(dim), # 对输入进行层归一化
        nn.Linear(dim, inner_dim, bias=False), # 线性变换，扩展维度
        nn.GELU(), # 应用GELU激活函数 
        nn.Linear(inner_dim, dim, bias=False), # 线性变换，恢复原始维度
    )
    

# ===========================
# 张量重塑函数
# ===========================

def reshape_tensor(x, heads):
    """
    重塑输入张量以适应多头注意力机制。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, length, width)。
        heads (int): 多头注意力的头数。

    Returns:
        torch.Tensor: 重塑后的张量，形状为 (batch_size * heads, length, dim_per_head)。
    """
    # 分离输入张量的维度：批次大小、长度和宽度
    bs, length, width = x.shape

    # 将张量重塑为 (batch_size, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)

    # 转置维度为 (batch_size, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)

    # 重新调整形状为 (batch_size * n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)

    return x


# ===========================
# PerceiverAttention 类
# ===========================

class PerceiverAttention(nn.Module):
    """
    Perceiver模型的注意力机制实现。

    该类实现了Perceiver模型中的注意力机制，允许模型处理高维输入（如图像），
    并通过交叉注意力机制与潜在表示（latents）进行交互。

    Args:
        dim (int): 输入和潜在表示的维度大小。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为8。
    """
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        # 缩放因子，用于缩放注意力得分
        self.scale = dim_head**-0.5
        # 每个注意力头的维度大小
        self.dim_head = dim_head
        # 多头注意力的头数
        self.heads = heads
        # 计算内部维度
        inner_dim = dim_head * heads

        # 定义层归一化层
        # 对输入进行层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 对潜在表示进行层归一化
        self.norm2 = nn.LayerNorm(dim)

        # 定义线性变换层，用于计算查询（Q）、键（K）和值（V）
        self.to_q = nn.Linear(dim, inner_dim, bias=False) # 查询的线性变换
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False) # 键和值的线性变换
        self.to_out = nn.Linear(inner_dim, dim, bias=False) # 输出线性变换


    def forward(self, x, latents):
        """
        前向传播方法，执行Perceiver的注意力机制计算。

        Args:
            x (torch.Tensor): 输入的图像特征，形状为 (batch_size, n1, D)。
            latents (torch.Tensor): 潜在表示特征，形状为 (batch_size, n2, D)。

        Returns:
            torch.Tensor: 经过注意力机制处理后的输出张量，形状为 (batch_size, n2, D)。
        """
        # 对输入张量进行层归一化
        x = self.norm1(x)
        # 对潜在表示张量进行层归一化
        latents = self.norm2(latents)
        
        # 获取潜在表示的批次大小和长度
        b, l, _ = latents.shape

        # 计算查询（Q）向量
        q = self.to_q(latents)
        # 将输入和潜在表示连接起来，作为键（K）和值（V）的输入
        kv_input = torch.cat((x, latents), dim=-2)
        # 计算键（K）和值（V）向量
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        # 重塑查询、键和值张量以适应多头注意力机制
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # 计算注意力得分
        # 使用缩放因子进行缩放，以确保梯度稳定
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # 计算注意力权重
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        # 应用softmax函数，将注意力权重转换为概率分布
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 通过注意力权重对值进行加权求和，得到输出
        out = weight @ v
        
        # 重塑输出张量的形状为 (batch_size, n_heads, length, dim_per_head)
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        # 通过线性变换层进行输出投影
        return self.to_out(out)


class Resampler(nn.Module):
    """
    重采样器（Resampler）类，用于在特征空间中执行重采样操作。

    该类通过使用潜在表示（latents）和多头注意力机制，从输入特征中提取和重采样信息。
    它可以用于例如图像特征的上采样、下采样或跨尺度特征融合等任务。

    Args:
        dim (int, optional): 潜在表示和注意力层的维度大小，默认为1024。
        depth (int, optional): 注意力层和前馈层的堆叠深度，默认为8。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为16。
        num_queries (int, optional): 查询（queries）的数量，默认为8。
        embedding_dim (int, optional): 输入嵌入特征的维度大小，默认为768。
        output_dim (int, optional): 输出特征的维度大小，默认为1024。
        ff_mult (int, optional): 前馈层内部维度相对于输入维度的倍数，默认为4。
    """
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        # 初始化潜在表示（latents），形状为 (1, num_queries, dim)
        # 使用正态分布初始化，并进行缩放以保持初始化的稳定性
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        # 定义一个线性变换层，用于将输入嵌入特征的维度从 embedding_dim 转换为 dim
        self.proj_in = nn.Linear(embedding_dim, dim)

        # 定义一个线性变换层，用于将潜在表示的维度从 dim 转换为 output_dim
        self.proj_out = nn.Linear(dim, output_dim)
        # 定义一个层归一化层，用于对输出进行归一化
        self.norm_out = nn.LayerNorm(output_dim)
        
        # 定义一个包含多个注意力层和前馈层的列表，用于堆叠多个处理层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 对于每一层，添加一个包含注意力层和前馈层的模块列表
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), # 多头注意力层
                        FeedForward(dim=dim, mult=ff_mult), # 前馈神经网络层
                    ]
                )
            )

    def forward(self, x):
        """
        前向传播方法，执行重采样操作。

        Args:
            x (torch.Tensor): 输入嵌入特征，形状为 (batch_size, sequence_length, embedding_dim)。

        Returns:
            torch.Tensor: 经过重采样后的输出特征，形状为 (batch_size, num_queries, output_dim)。
        """
        # 重复潜在表示，以匹配输入的批次大小，形状变为 (batch_size, num_queries, dim)
        latents = self.latents.repeat(x.size(0), 1, 1)

        # 将输入嵌入特征的维度从 embedding_dim 转换为 dim
        x = self.proj_in(x)
        
        # 遍历每一层，包括注意力层和前馈层
        for attn, ff in self.layers:
            # 执行注意力机制，将输入特征与潜在表示结合，更新潜在表示
            latents = attn(x, latents) + latents
            # 通过前馈神经网络进一步处理潜在表示
            latents = ff(latents) + latents
        
        # 将潜在表示的维度从 dim 转换为 output_dim
        latents = self.proj_out(latents)
        # 对输出进行层归一化
        return self.norm_out(latents)

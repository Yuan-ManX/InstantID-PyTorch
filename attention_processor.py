import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e:
    xformers_available = False


# 定义一个RegionControler类，用于控制区域相关的功能
class RegionControler(object):
    def __init__(self) -> None:
        """
        初始化RegionControler实例。
        
        Attributes:
            prompt_image_conditioning (list): 用于存储与图像提示相关的条件列表。
        """
        # 初始化一个空列表，用于存储图像提示条件
        self.prompt_image_conditioning = []

# 创建RegionControler的实例
region_control = RegionControler()


# 定义一个继承自nn.Module的AttnProcessor类，用于处理注意力机制相关的计算
class AttnProcessor(nn.Module):
    """
    默认的处理器，用于执行与注意力机制相关的计算。
    
    该类实现了Transformer模型中的注意力机制，包括查询（Q）、键（K）、值（V）的计算，
    以及注意力权重的计算和应用的整个流程。
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        """
        初始化AttnProcessor实例。
        
        Args:
            hidden_size (int, optional): 隐藏层的维度大小。如果提供，将用于定义线性变换层。
            cross_attention_dim (int, optional): 跨注意力机制的维度大小。如果提供，将用于跨注意力计算。
        """
        super().__init__()

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """
        前向传播方法，执行注意力机制的计算。
        
        Args:
            attn: 注意力机制相关的配置和参数。
            hidden_states (torch.Tensor): 输入的隐藏状态张量，形状通常为 (batch_size, sequence_length, hidden_size)。
            encoder_hidden_states (torch.Tensor, optional): 编码器输出的隐藏状态张量，用于跨注意力机制。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置以防止模型关注这些位置。
            temb (torch.Tensor, optional): 时间步张量，用于条件生成等任务。
        
        Returns:
            torch.Tensor: 经过注意力机制处理后的隐藏状态张量。
        """
        # 保留输入的隐藏状态作为残差连接
        residual = hidden_states

        # 如果存在空间归一化，则应用空间归一化
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 获取输入张量的维度
        input_ndim = hidden_states.ndim

        # 如果输入是4维张量（通常用于图像数据），则将其重塑为2维张量
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # 将高度和宽度维度合并，并转置为 (batch_size, height*width, channel)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 获取批次大小和序列长度
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # 准备注意力掩码，确保其形状与注意力计算兼容
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 如果存在组归一化，则应用组归一化
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算查询（Q）向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，则将隐藏状态作为编码器隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要跨注意力归一化，则对编码器隐藏状态进行归一化
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 计算键（K）和值（V）向量
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 将查询、键和值向量的维度从 (batch_size, sequence_length, hidden_size) 转换为 (batch_size*num_heads, sequence_length, head_dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算注意力得分（注意力权重）
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 通过注意力权重对值向量进行加权求和，得到新的隐藏状态
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态从 (batch_size*num_heads, sequence_length, head_dim) 转换回 (batch_size, sequence_length, hidden_size)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 通过线性投影层进行线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用Dropout正则化
        hidden_states = attn.to_out[1](hidden_states)

        # 如果输入是4维张量，则将其重塑回原始形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 应用残差连接，将原始隐藏状态与经过注意力机制处理的隐藏状态相加
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 对输出进行缩放
        hidden_states = hidden_states / attn.rescale_output_factor

        # 返回最终的隐藏状态
        return hidden_states
    

class IPAttnProcessor(nn.Module):
    """
    IP-Adapter 的注意力处理器。
    
    该处理器扩展了标准的注意力机制，专门用于处理图像提示（Image Prompt, IP），
    以增强模型在图像相关任务中的表现。
    
    Args:
        hidden_size (int): 
            注意力层的隐藏层大小。
        cross_attention_dim (int, optional): 
            `encoder_hidden_states` 的通道数。如果未提供，则默认为 `hidden_size`。
        scale (float, optional, default=1.0): 
            图像提示的权重缩放因子。
        num_tokens (int, optional, default=4): 
            图像特征的上下文长度。当使用 `ip_adapter_plus` 时，应设置为16。
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        """
        初始化 IPAttnProcessor 实例。
        
        Args:
            hidden_size (int): 注意力层的隐藏层大小。
            cross_attention_dim (int, optional): `encoder_hidden_states` 的通道数。如果未提供，则默认为 `hidden_size`。
            scale (float, optional, default=1.0): 图像提示的权重缩放因子。
            num_tokens (int, optional, default=4): 图像特征的上下文长度。当使用 `ip_adapter_plus` 时，应设置为16。
        """
        super().__init__()
        
        # 初始化参数
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 跨注意力维度
        self.cross_attention_dim = cross_attention_dim
        # 图像提示的权重缩放因子
        self.scale = scale
        # 图像特征的上下文长度
        self.num_tokens = num_tokens

        # 定义用于处理图像提示的线性变换层
        # 如果 cross_attention_dim 未提供，则使用 hidden_size 作为输入维度
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) # 键（Key）的线性变换层
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) # 值（Value）的线性变换层

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """
        前向传播方法，执行扩展的注意力机制计算，包括图像提示的处理。
        
        Args:
            attn: 注意力机制相关的配置和参数。
            hidden_states (torch.Tensor): 输入的隐藏状态张量，形状通常为 (batch_size, sequence_length, hidden_size)。
            encoder_hidden_states (torch.Tensor, optional): 编码器输出的隐藏状态张量，用于跨注意力机制。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置以防止模型关注这些位置。
            temb (torch.Tensor, optional): 时间步张量，用于条件生成等任务。
        
        Returns:
            torch.Tensor: 经过扩展的注意力机制处理后的隐藏状态张量。
        """
        # 保留输入的隐藏状态作为残差连接
        residual = hidden_states

        # 如果存在空间归一化，则应用空间归一化
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 获取输入张量的维度
        input_ndim = hidden_states.ndim

        # 如果输入是4维张量（通常用于图像数据），则将其重塑为2维张量
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # 将高度和宽度维度合并，并转置为 (batch_size, height*width, channel)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 获取批次大小和序列长度
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # 准备注意力掩码，确保其形状与注意力计算兼容
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 如果存在组归一化，则应用组归一化
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算查询（Q）向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，则将隐藏状态作为编码器隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # 分离编码器隐藏状态和图像提示隐藏状态
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            # 如果需要跨注意力归一化，则对编码器隐藏状态进行归一化
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 计算键（K）和值（V）向量
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 将查询、键和值向量的维度从 (batch_size, sequence_length, hidden_size) 转换为 (batch_size*num_heads, sequence_length, head_dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 使用高效的注意力机制实现（xformers）
        if xformers_available:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        else:
            # 计算注意力得分（注意力权重）
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # 通过注意力权重对值向量进行加权求和，得到新的隐藏状态
            hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态从 (batch_size*num_heads, sequence_length, head_dim) 转换回 (batch_size, sequence_length, hidden_size)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # 处理图像提示的键和值
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        
        # 将图像提示的键和值向量的维度转换为 (batch_size*num_heads, num_tokens, head_dim)
        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)
        
        # 对图像提示应用注意力机制
        if xformers_available:
            ip_hidden_states = self._memory_efficient_attention_xformers(query, ip_key, ip_value, None)
        else:
            # 计算图像提示的注意力得分
            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            # 通过注意力权重对图像提示的值向量进行加权求和
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        # 将图像提示的隐藏状态转换为 (batch_size, num_tokens, hidden_size)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        # 区域控制：应用区域掩码
        if len(region_control.prompt_image_conditioning) == 1:
            region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
            if region_mask is not None:
                h, w = region_mask.shape[:2]
                # 计算缩放比例，并调整区域掩码的尺寸以匹配查询（query）的长度
                ratio = (h * w / query.shape[1]) ** 0.5
                mask = F.interpolate(region_mask[None, None], scale_factor=1/ratio, mode='nearest').reshape([1, -1, 1])
            else:
                mask = torch.ones_like(ip_hidden_states)
            # 应用区域掩码到图像提示的隐藏状态
            ip_hidden_states = ip_hidden_states * mask     

        # 将图像提示的隐藏状态与主隐藏状态结合
        hidden_states = hidden_states + self.scale * ip_hidden_states

        # 通过线性投影层进行线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用Dropout正则化
        hidden_states = attn.to_out[1](hidden_states)

        # 如果输入是4维张量，则将其重塑回原始形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 应用残差连接，将原始隐藏状态与经过注意力机制处理的隐藏状态相加
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 对输出进行缩放
        hidden_states = hidden_states / attn.rescale_output_factor

        # 返回最终的隐藏状态
        return hidden_states


    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        """
        使用 xformers 库的高效注意力机制实现。
        
        Args:
            query (torch.Tensor): 查询张量。
            key (torch.Tensor): 键张量。
            value (torch.Tensor): 值张量。
            attention_mask (torch.Tensor, optional): 注意力掩码张量。
        
        Returns:
            torch.Tensor: 经过注意力机制处理后的隐藏状态张量。
        """
        # 确保张量是连续的
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        # 调用 xformers 的高效注意力机制实现
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        # 如果需要，可以在这里添加对隐藏状态的进一步处理，例如重塑维度
        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    """
    使用缩放点积注意力（scaled dot-product attention）的处理器。
    如果您使用的是PyTorch 2.0及以上版本，则默认启用此功能。
    
    该处理器利用PyTorch 2.0中引入的`scaled_dot_product_attention`函数，
    提供了一种更高效且简化的注意力机制实现。
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        """
        初始化AttnProcessor2_0实例。
        
        Args:
            hidden_size (int, optional): 隐藏层的维度大小。如果提供，将用于定义线性变换层。
            cross_attention_dim (int, optional): 跨注意力机制的维度大小。如果提供，将用于跨注意力计算。
        
        Raises:
            ImportError: 如果当前PyTorch版本低于2.0，则抛出导入错误，提示需要升级PyTorch。
        """
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """
        前向传播方法，执行缩放点积注意力计算。
        
        Args:
            attn: 包含注意力机制相关配置和参数的实例。
            hidden_states (torch.Tensor): 输入的隐藏状态张量，形状通常为 (batch_size, sequence_length, hidden_size)。
            encoder_hidden_states (torch.Tensor, optional): 编码器输出的隐藏状态张量，用于跨注意力机制。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置以防止模型关注这些位置。
            temb (torch.Tensor, optional): 时间步张量，用于条件生成等任务。
        
        Returns:
            torch.Tensor: 经过缩放点积注意力处理后的隐藏状态张量。
        """
        # 保留输入的隐藏状态作为残差连接
        residual = hidden_states

        # 如果存在空间归一化，则应用空间归一化
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 获取输入张量的维度
        input_ndim = hidden_states.ndim

        # 如果输入是4维张量（通常用于图像数据），则将其重塑为2维张量
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # 将高度和宽度维度合并，并转置为 (batch_size, height*width, channel)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 获取批次大小和序列长度
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # 如果提供了注意力掩码，则进行处理
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention 期望的注意力掩码形状为 (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # 如果存在组归一化，则应用组归一化
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算查询（Q）向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，则将隐藏状态作为编码器隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要跨注意力归一化，则对编码器隐藏状态进行归一化
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 计算键（K）和值（V）向量
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 获取键的内部维度
        inner_dim = key.shape[-1]
        # 计算每个头部的维度大小
        head_dim = inner_dim // attn.heads

        # 重塑查询、键和值张量，以适应多头注意力的计算
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 使用PyTorch 2.0的缩放点积注意力函数计算注意力
        # hidden_states 的形状为 (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # 将隐藏状态重塑回 (batch, seq_len, num_heads * head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 通过线性投影层进行线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用Dropout正则化
        hidden_states = attn.to_out[1](hidden_states)

        # 如果输入是4维张量，则将其重塑回原始形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 应用残差连接，将原始隐藏状态与经过注意力机制处理的隐藏状态相加
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 对输出进行缩放
        hidden_states = hidden_states / attn.rescale_output_factor

        # 返回最终的隐藏状态
        return hidden_states


class IPAttnProcessor2_0(torch.nn.Module):
    """
    针对PyTorch 2.0的IP-Adapter的注意力处理器。
    
    该处理器扩展了标准的缩放点积注意力机制，专门用于处理图像提示（Image Prompt, IP），
    以增强模型在图像相关任务中的表现。
    
    Args:
        hidden_size (int): 
            注意力层的隐藏层大小。
        cross_attention_dim (int, optional): 
            `encoder_hidden_states` 的通道数。如果未提供，则默认为 `hidden_size`。
        scale (float, optional, default=1.0): 
            图像提示的权重缩放因子。
        num_tokens (int, optional, default=4): 
            图像特征的上下文长度。当使用 `ip_adapter_plus` 时，应设置为16。
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        """
        初始化 IPAttnProcessor2_0 实例。
        
        Args:
            hidden_size (int): 注意力层的隐藏层大小。
            cross_attention_dim (int, optional): `encoder_hidden_states` 的通道数。如果未提供，则默认为 `hidden_size`。
            scale (float, optional, default=1.0): 图像提示的权重缩放因子。
            num_tokens (int, optional, default=4): 图像特征的上下文长度。当使用 `ip_adapter_plus` 时，应设置为16。
        """
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        # 初始化参数
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 跨注意力维度
        self.cross_attention_dim = cross_attention_dim
        # 图像提示的权重缩放因子
        self.scale = scale
        # 图像特征的上下文长度
        self.num_tokens = num_tokens

        # 定义用于处理图像提示的线性变换层
        # 如果 cross_attention_dim 未提供，则使用 hidden_size 作为输入维度
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) # 键（Key）的线性变换层
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) # 值（Value）的线性变换层

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        """
        前向传播方法，执行扩展的缩放点积注意力计算，包括图像提示的处理。
        
        Args:
            attn: 包含注意力机制相关配置和参数的实例。
            hidden_states (torch.Tensor): 输入的隐藏状态张量，形状通常为 (batch_size, sequence_length, hidden_size)。
            encoder_hidden_states (torch.Tensor, optional): 编码器输出的隐藏状态张量，用于跨注意力机制。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置以防止模型关注这些位置。
            temb (torch.Tensor, optional): 时间步张量，用于条件生成等任务。
        
        Returns:
            torch.Tensor: 经过扩展的缩放点积注意力处理后的隐藏状态张量。
        """
        # 保留输入的隐藏状态作为残差连接
        residual = hidden_states

        # 如果存在空间归一化，则应用空间归一化
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        # 获取输入张量的维度
        input_ndim = hidden_states.ndim

        # 如果输入是4维张量（通常用于图像数据），则将其重塑为2维张量
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # 将高度和宽度维度合并，并转置为 (batch_size, height*width, channel)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 获取批次大小和序列长度
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # 如果提供了注意力掩码，则进行处理
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention 期望的注意力掩码形状为 (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # 如果存在组归一化，则应用组归一化
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算查询（Q）向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，则将隐藏状态作为编码器隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # 分离编码器隐藏状态和图像提示隐藏状态
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            # 如果需要跨注意力归一化，则对编码器隐藏状态进行归一化
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 计算键（K）和值（V）向量
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 获取键的内部维度
        inner_dim = key.shape[-1]
        # 计算每个头部的维度大小
        head_dim = inner_dim // attn.heads

        # 重塑查询、键和值张量，以适应多头注意力的计算
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 使用PyTorch 2.0的缩放点积注意力函数计算注意力
        # hidden_states 的形状为 (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # 将隐藏状态重塑回 (batch, seq_len, num_heads * head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 处理图像提示的键和值
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        # 重塑图像提示的键和值张量，以适应多头注意力的计算
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 对图像提示应用缩放点积注意力
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # 计算注意力映射（可选）
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # 打印注意力映射的形状（可选）
            # print(self.attn_map.shape)
        
        # 重塑图像提示的隐藏状态回 (batch, seq_len, hidden_size)
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        # 区域控制：应用区域掩码
        if len(region_control.prompt_image_conditioning) == 1:
            region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
            if region_mask is not None:
                query = query.reshape([-1, query.shape[-2], query.shape[-1]])
                h, w = region_mask.shape[:2]
                ratio = (h * w / query.shape[1]) ** 0.5
                mask = F.interpolate(region_mask[None, None], scale_factor=1/ratio, mode='nearest').reshape([1, -1, 1])
            else:
                mask = torch.ones_like(ip_hidden_states)
            ip_hidden_states = ip_hidden_states * mask

        # 将图像提示的隐藏状态与主隐藏状态结合
        hidden_states = hidden_states + self.scale * ip_hidden_states

        # 通过线性投影层进行线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用Dropout正则化
        hidden_states = attn.to_out[1](hidden_states)

        # 如果输入是4维张量，则将其重塑回原始形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 应用残差连接，将原始隐藏状态与经过注意力机制处理的隐藏状态相加
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 对输出进行缩放
        hidden_states = hidden_states / attn.rescale_output_factor

        # 返回最终的隐藏状态
        return hidden_states

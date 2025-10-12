import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAttention(nn.Module):
    """分离注意力模块 - 用于增强特征多样性"""
    
    def __init__(self, channels=256, radix=2):
        super().__init__()
        self.channels = channels
        self.radix = radix
        self.split_channels = channels // radix  # 128
        
        # 升维卷积 - 保持Memory Attention接口不变
        self.temporal_upconv = nn.Conv2d(self.split_channels, channels, 1)
        self.modal_upconv = nn.Conv2d(self.split_channels, channels, 1)
        
#         # Split权重生成 - 学习如何平衡两种增强
#         self.split_weight_gen = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // 8, 1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 8, radix, 1),  # 2个split的权重
#             nn.Softmax(dim=1)
#         )
        
    def forward(self, x, temporal_memory_module=None, modal_memory_module=None, 
                temporal_kwargs=None, modal_kwargs=None):
        """
        Args:
            x: [B, 256, 64, 64] backbone特征
            temporal_memory_module: 帧间Memory Attention模块
            modal_memory_module: 模态间Memory Attention模块
            temporal_kwargs: 帧间memory attention的参数字典
            modal_kwargs: 模态间memory attention的参数字典
        """
        B, C, H, W = x.shape
        
        # 设置默认参数
        if temporal_kwargs is None:
            temporal_kwargs = {}
        if modal_kwargs is None:
            modal_kwargs = {}
        
        # 1. 按通道分割特征
        temporal_feat, modal_feat = torch.split(x, self.split_channels, dim=1)
        
        # 2. 升维到Memory Attention需要的维度
        temporal_input = self.temporal_upconv(temporal_feat)  # [B, 256, 64, 64]
        modal_input = self.modal_upconv(modal_feat)           # [B, 256, 64, 64]
        
        # 3. 转换为Memory Attention需要的格式 [HW, B, C]
        temporal_input_flat = temporal_input.flatten(2).permute(2, 0, 1)  # [4096, B, 256]
        modal_input_flat = modal_input.flatten(2).permute(2, 0, 1)        # [4096, B, 256]
        
        # 4. Memory Attention处理
        if temporal_memory_module is not None and temporal_kwargs:
            # 使用帧间Memory Attention
            enhanced_temporal_flat = temporal_memory_module(
                curr=[temporal_input_flat],
                **temporal_kwargs
            )
            # 转换回 [B, C, H, W] 格式
            enhanced_temporal = enhanced_temporal_flat.permute(1, 2, 0).view(B, C, H, W)
        else:
            enhanced_temporal = temporal_input
            
        if modal_memory_module is not None and modal_kwargs:
            # 使用模态间Memory Attention
            enhanced_modal_flat = modal_memory_module(
                curr=[modal_input_flat],
                **modal_kwargs
            )
            # 转换回 [B, C, H, W] 格式
            enhanced_modal = enhanced_modal_flat.permute(1, 2, 0).view(B, C, H, W)
        else:
            enhanced_modal = modal_input
            
        # 5. 生成自适应权重
#         split_weights = self.split_weight_gen(x)  # [B, 2, 1, 1]
#         temporal_weight = split_weights[:, 0:1, :, :]  # [B, 1, 1, 1]
#         modal_weight = split_weights[:, 1:2, :, :]     # [B, 1, 1, 1]
        
#         # 6. 加权融合
#         enhanced_features = (
#             temporal_weight * enhanced_temporal + 
#             modal_weight * enhanced_modal
#             + x # 残差连接保留原始特征
#         )  # [B, 256, 64, 64]
        enhanced_features = enhanced_temporal + enhanced_modal
        return enhanced_features

def dummy_temporal_memory_attention(x):
    """模拟帧间Memory Attention"""
    print(f"  [帧间Memory] 输入: {x.shape} -> 输出: {x.shape}")
    return x

def dummy_modal_memory_attention(x):
    """模拟模态间Memory Attention"""
    print(f"  [模态间Memory] 输入: {x.shape} -> 输出: {x.shape}")
    return x

def test_simple_split_attention():
    """测试简单分离注意力"""
    print("=" * 60)
    print("测试简单分离注意力 (Channel Split)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SplitAttention(channels=256).to(device)
    
    # 模拟backbone特征输入 (来自ViT)
    batch_size = 2
    vit_features = torch.randn(batch_size, 256, 64, 64).to(device)
    
    print(f"\n测试输入:")
    print(f"Batch size: {batch_size}")
    print(f"ViT特征: {vit_features.shape}")
    
    # 测试1: placeholder模式
    print(f"\n" + "-" * 40)
    print("测试1: 使用placeholder Memory Attention")
    print("-" * 40)
    
    with torch.no_grad():
        output1 = model(vit_features)
        
    print(f"\n测试1结果:")
    print(f"输入维度: {vit_features.shape}")
    print(f"输出维度: {output1.shape}")
    print(f"维度匹配: {vit_features.shape == output1.shape}")
    
    # 测试2: dummy函数模式
    print(f"\n" + "-" * 40)
    print("测试2: 使用dummy Memory Attention函数")
    print("-" * 40)
    
    with torch.no_grad():
        output2 = model(
            vit_features,
            temporal_memory_fn=dummy_temporal_memory_attention,
            modal_memory_fn=dummy_modal_memory_attention
        )
        
    print(f"\n测试2结果:")
    print(f"输入维度: {vit_features.shape}")
    print(f"输出维度: {output2.shape}")
    print(f"维度匹配: {vit_features.shape == output2.shape}")
    
    # 验证
    print(f"\n" + "-" * 40)
    print("验证结果")
    print("-" * 40)
    print(f"两次输出相同: {torch.allclose(output1, output2)}")
    print(f"输出数值范围: [{output1.min():.4f}, {output1.max():.4f}]")
    
    # 模型信息
    print(f"\n" + "-" * 40)
    print("模型信息")
    print("-" * 40)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    print(f"\n" + "=" * 60)
    print("测试完成! 简单分离策略更适合ViT特征")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_split_attention()
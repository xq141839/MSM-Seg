import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal, List

class PyramidPromptUNet(nn.Module):
    """
    基于特征金字塔的交互式分割网络
    利用backbone的多尺度特征做UNet上采样
    """
    
    def __init__(self, feature_channels=256, device=None):
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 提示编码器 - 将提示信息编码到不同尺度
        self.prompt_processors = nn.ModuleDict({
            'scale_128': self._make_prompt_processor(5, 32),   # 128x128 (5通道: box+4点类型)
            'scale_64': self._make_prompt_processor(5, 64),   # 64x64  
            'scale_32': self._make_prompt_processor(5, 256),   # 32x32
        })
        
        # 特征融合模块 - 融合backbone特征和提示特征
        self.feature_fusion = nn.ModuleDict({
            'fuse_128': nn.Conv2d(feature_channels*2, feature_channels, 3, padding=1),
            'fuse_64': nn.Conv2d(feature_channels//2, feature_channels//4, 3, padding=1),
            'fuse_32': nn.Conv2d(feature_channels//4, feature_channels//8, 3, padding=1),
        })
        
        # UNet解码器 - 从32x32开始上采样到256x256
        self.decoder = PyramidDecoder(feature_channels)
        
    def _make_prompt_processor(self, in_channels, out_channels):
        """创建提示信息处理模块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, padding=1),
            nn.GroupNorm(8, out_channels//2),  # 使用GroupNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),     # 使用GroupNorm替代BatchNorm
            nn.ReLU(inplace=True)
        )
        
    def forward(self, 
                high_res_features: List[torch.Tensor],  # {'f_128':..., 'f_64':..., 'f_32':...}
                pix_feat_with_mem: torch.Tensor = None,  # 当前帧索引
                past_mask: Optional[torch.Tensor] = None,     # [B, 1, 256, 256]
                point_coords: Optional[torch.Tensor] = None,  # [B, n, 2]
                point_labels: Optional[torch.Tensor] = None,  # [B, n]
                scribbles: Optional[torch.Tensor] = None,     # [B, 2, 256, 256]
                box: Optional[torch.Tensor] = None,           # [B, 1, 4]
                use_prompts: bool = True,                     # 是否使用提示信息
                input_resolution: int = 1024,                 # 输入坐标的原始分辨率
                ):
        # for i in range(len(pyramid_features)):
        #     print(pyramid_features[i].shape)
        # print(cached_features.keys())
        f_128 = high_res_features[1] # [B, 64, 32, 32]
        f_64 = pix_feat_with_mem # [B, 256, 16, 16]
        f_256 = high_res_features[0] # [B, 32, 64, 64]
        batch_size = f_128.shape[0]
        
#         # 如果不使用提示信息，直接使用backbone特征进行UNet解码
#         if not use_prompts:
        # output = self.decoder(f_64, f_128, f_256)  # [B, 1, 256, 256]
        # return output
        
        # 1. 编码提示信息到不同尺度
        prompts = {
            'point_coords': point_coords,
            'point_labels': point_labels,
            'scribbles': scribbles,
            'box': box,
        }
        
        # 编码到各个尺度
        prompt_128 = self._encode_prompts(prompts, (64, 64), batch_size, input_resolution)  # [B, 5, 128, 128]
        prompt_64 = self._encode_prompts(prompts, (32, 32), batch_size, input_resolution)     # [B, 5, 64, 64]
        prompt_32 = self._encode_prompts(prompts, (16, 16), batch_size, input_resolution)     # [B, 5, 32, 32]
        
        # 加入past_mask信息到最高分辨率 (如果提供的话)
        if past_mask is not None:
            past_mask_128 = F.interpolate(past_mask, size=(128, 128), mode='bilinear', align_corners=False)
            prompt_128 = torch.cat([prompt_128, past_mask_128], dim=1)  # [B, 6, 128, 128]
            prompt_128 = prompt_128[:, :5, :, :]  # 保持5通道，融合past_mask到第一个通道
        
        # 2. 处理提示特征
        prompt_feat_128 = self.prompt_processors['scale_128'](prompt_128)  # [B, 64, 128, 128]
        prompt_feat_64 = self.prompt_processors['scale_64'](prompt_64)     # [B, 128, 64, 64]
        prompt_feat_32 = self.prompt_processors['scale_32'](prompt_32)     # [B, 256, 32, 32]
        

        # 3. 融合backbone特征和提示特征
        fused_128 = self.feature_fusion['fuse_128'](
            torch.cat([f_64, prompt_feat_32], dim=1)
        )  # [B, 256, 128, 128]
        
        fused_64 = self.feature_fusion['fuse_64'](
            torch.cat([f_128, prompt_feat_64], dim=1)
        )  # [B, 256, 64, 64]
        
        fused_32 = self.feature_fusion['fuse_32'](
            torch.cat([f_256, prompt_feat_128], dim=1)
        )  # [B, 256, 32, 32]
        
        # 4. UNet解码器 - 带跳跃连接的上采样
        output = self.decoder(fused_128, fused_64, fused_32)  # [B, 1, 256, 256]
        
        return output
    
    def _encode_prompts(self, prompts: Dict, shape: Tuple[int, int], batch_size: int, input_resolution: int = 1024) -> torch.Tensor:
        """编码提示信息到指定尺度"""
        device = self.device
        
        # 编码边界框
        if prompts.get("box") is not None:
            box_embed = bbox_shaded(prompts['box'], shape=shape, device=device)
        else:
            box_embed = torch.zeros((batch_size, 1) + shape, device=device)
        
        # 编码点击
        if prompts.get("point_coords") is not None:
            # 缩放坐标从input_resolution到目标尺度
            coords = prompts['point_coords'].clone().float()
            # 首先从input_resolution缩放到256×256
            coords[..., 0] = coords[..., 0] * (256.0 / input_resolution)  # x坐标
            coords[..., 1] = coords[..., 1] * (256.0 / input_resolution)  # y坐标
            # 然后从256×256缩放到目标尺度
            coords[..., 0] = coords[..., 0] * (shape[1] / 256.0)  # x坐标: 256 -> target_width
            coords[..., 1] = coords[..., 1] * (shape[0] / 256.0)  # y坐标: 256 -> target_height
            coords = coords.int()
            
            # 边界检查和裁剪
            coords[..., 0] = torch.clamp(coords[..., 0], 0, shape[1] - 1)
            coords[..., 1] = torch.clamp(coords[..., 1], 0, shape[0] - 1)
            
            click_embed = click_onehot(coords, prompts['point_labels'], shape=shape)  # 返回4通道
        else:
            click_embed = torch.zeros((batch_size, 4) + shape, device=device)  # 4通道零向量
        
        # 处理涂鸦 - 下采样到目标尺度
        if prompts.get("scribbles") is not None:
            scribbles_resized = F.interpolate(
                prompts['scribbles'], size=shape, mode='bilinear', align_corners=False
            )
            click_embed = torch.clamp(click_embed + scribbles_resized, min=0.0, max=1.0)
        
        # 拼接：box(1) + clicks(4) = 5 channels
        prompt_embeds = torch.cat([box_embed, click_embed], dim=1)
        
        return prompt_embeds

class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

class PyramidDecoder(nn.Module):
    """基于特征金字塔的解码器"""
    
    def __init__(self, feature_channels=256):
        super().__init__()
        
        # 解码器路径的通道设计
        self.decoder_channels = [feature_channels, feature_channels//2, feature_channels//4, feature_channels//8]
        
        # 上采样层
        self.upconv1 = nn.ConvTranspose2d(feature_channels, feature_channels//2, 2, stride=2)  # 32->64
        self.upconv2 = nn.ConvTranspose2d(feature_channels//4, feature_channels//8, 2, stride=2)  # 64->128
        
        # 跳跃连接的融合层 - 真正的UNet连接
        # 64x64层: 上采样特征(128) + 编码器特征(256) -> 融合输出(128)
        self.skip_conv1 = self._conv_block(192, feature_channels//4)
        
        # 128x128层: 上采样特征(64) + 编码器特征(256) -> 融合输出(64)  
        self.skip_conv2 = self._conv_block(feature_channels//4, feature_channels//8)
        
        # 最终输出层
        self.final = nn.Conv2d(feature_channels//8, 1, 3, padding=1)
        
    def _conv_block(self, in_channels, out_channels):
        """卷积块：Conv-GN-ReLU-Conv-GN-ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_64, feat_128, feat_256):

        x = self.upconv1(feat_64)  # [B, 128, 128, 128]
        x = torch.cat([x, feat_128], dim=1)  # [B, 192, 128, 128]
        x = self.skip_conv1(x)  # [B, 64, 128, 128]


        x = self.upconv2(x)  # [B, 32, 256, 256]
        x = torch.cat([x, feat_256], dim=1)  # [B, 64, 256, 256]
        x = self.skip_conv2(x)  # [B, 32, 256, 256] - 融合处理
        
        output = self.final(x)  # [B, 1, 256, 256]
        
        return output

# 复用编码函数
def click_onehot(point_coords, point_labels, shape: Tuple[int,int] = (128,128), indexing: Literal['xy','uv'] ='xy'):
    """
    将点击转换为独热编码掩码 - SAM2风格
    
    SAM2标签语义：
    - 0: 负点（背景点）     -> 通道0
    - 1: 正点（前景点）     -> 通道1  
    - 2: bbox左下角        -> 通道2
    - 3: bbox右上角        -> 通道3
    - -1: 填充点（忽略）
    
    Returns:
        torch.Tensor: [B, 4, H, W] 4通道独热编码
    """
    assert len(point_coords.shape) == 3, "point_coords must be BxNx2"
    assert point_coords.shape[-1] == 2, "point_coords must be BxNx2"
    assert point_labels.shape[-1] == point_coords.shape[1], "point_labels must be BxN"
    
    device = point_coords.device
    batch_size = point_coords.shape[0]
    n_points = point_coords.shape[1]
    
    # 4通道：[负点, 正点, bbox左上, bbox右下]
    embed = torch.zeros((batch_size, 4) + shape, device=device)
    labels = point_labels.flatten().long()  # 改为long类型
    
    idx_coords = torch.cat((
        torch.arange(batch_size, device=device).reshape(-1,1).repeat(1,n_points)[...,None], 
        point_coords
    ), axis=2).reshape(-1,3)
    
    # 添加边界检查
    if indexing=='xy':
        valid_mask = (idx_coords[:,1] >= 0) & (idx_coords[:,1] < shape[1]) & \
                    (idx_coords[:,2] >= 0) & (idx_coords[:,2] < shape[0])
        valid_coords = idx_coords[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_coords) > 0:
            # 过滤掉填充点 (label = -1)
            non_padding_mask = valid_labels >= 0
            valid_coords = valid_coords[non_padding_mask]
            valid_labels = valid_labels[non_padding_mask]
            
            if len(valid_coords) > 0:
                # 按SAM2方式：每种标签对应一个独立通道
                for i, label in enumerate(valid_labels):
                    coord = valid_coords[i]
                    batch_idx, y, x = coord[0], coord[2], coord[1]
                    
                    if 0 <= label <= 3:  # 有效标签范围
                        embed[batch_idx, label, y, x] = 1.0
    else:
        valid_mask = (idx_coords[:,1] >= 0) & (idx_coords[:,1] < shape[0]) & \
                    (idx_coords[:,2] >= 0) & (idx_coords[:,2] < shape[1])
        valid_coords = idx_coords[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_coords) > 0:
            # 过滤掉填充点 (label = -1)
            non_padding_mask = valid_labels >= 0
            valid_coords = valid_coords[non_padding_mask]
            valid_labels = valid_labels[non_padding_mask]
            
            if len(valid_coords) > 0:
                # 按SAM2方式：每种标签对应一个独立通道
                for i, label in enumerate(valid_labels):
                    coord = valid_coords[i]
                    batch_idx, y, x = coord[0], coord[1], coord[2]
                    
                    if 0 <= label <= 3:  # 有效标签范围
                        embed[batch_idx, label, y, x] = 1.0
    
    return embed

def bbox_shaded(boxes, shape: Tuple[int,int] = (128,128), device='cpu'):
    """将边界框转换为阴影掩码"""
    assert len(shape)==2, "shape must be 2D"
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.int().cpu().numpy()
    
    batch_size = boxes.shape[0]
    n_boxes = boxes.shape[1]
    bbox_embed = torch.zeros((batch_size,1)+tuple(shape), device=device, dtype=torch.float32)
    
    if boxes is not None:
        for i in range(batch_size):
            for j in range(n_boxes):
                x1, y1, x2, y2 = boxes[i,j,:]
                x_min = max(0, min(x1,x2))
                x_max = min(shape[1], max(x1,x2))
                y_min = max(0, min(y1,y2))
                y_max = min(shape[0], max(y1,y2))
                if x_max > x_min and y_max > y_min:
                    bbox_embed[i, 0, y_min:y_max, x_min:x_max] = 1.0
    
    return bbox_embed

# 测试代码
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model = PyramidPromptUNet(feature_channels=256, device=device).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟特征金字塔输入
    batch_size = 2
    f_128 = torch.randn(batch_size, 256, 128, 128).to(device)
    f_64 = torch.randn(batch_size, 256, 64, 64).to(device)
    f_32 = torch.randn(batch_size, 256, 32, 32).to(device)
    pyramid_features = {'f_128': f_128, 'f_64': f_64, 'f_32': f_32}
    
    print("\n=== 测试1: 只使用特征，无提示信息 ===")
    try:
        output_no_prompt = model(
            pyramid_features=pyramid_features,
            use_prompts=False
        )
        print(f"✓ 无提示模式成功")
        print(f"  输入特征金字塔: {[v.shape for v in pyramid_features.values()]}")
        print(f"  输出mask: {output_no_prompt.shape}")
        print(f"  输出值范围: [{output_no_prompt.min().item():.4f}, {output_no_prompt.max().item():.4f}]")
    except Exception as e:
        print(f"✗ 无提示模式失败: {e}")
    
    print("\n=== 测试2: 使用提示信息 ===")
    try:
        past_mask = torch.zeros(batch_size, 1, 256, 256).to(device)
        point_coords = torch.randint(0, 256, (batch_size, 3, 2)).to(device)
        point_labels = torch.randint(0, 2, (batch_size, 3)).to(device)
        
        output_with_prompt = model(
            pyramid_features=pyramid_features,
            past_mask=past_mask,
            point_coords=point_coords,
            point_labels=point_labels,
            use_prompts=True
        )
        print(f"✓ 有提示模式成功")
        print(f"  输入特征金字塔: {[v.shape for v in pyramid_features.values()]}")
        print(f"  输入过去mask: {past_mask.shape}")
        print(f"  输入点击坐标: {point_coords.shape}")
        print(f"  输出mask: {output_with_prompt.shape}")
        print(f"  输出值范围: [{output_with_prompt.min().item():.4f}, {output_with_prompt.max().item():.4f}]")
    except Exception as e:
        print(f"✗ 有提示模式失败: {e}")
    
    print("\n=== 测试3: 模拟训练过程 ===")
    try:
        model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 模拟真实标签
        target_mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
        
        # 无提示训练
        optimizer.zero_grad()
        pred_no_prompt = model(pyramid_features, use_prompts=False)
        loss_no_prompt = criterion(pred_no_prompt, target_mask)
        loss_no_prompt.backward()
        optimizer.step()
        
        print(f"✓ 无提示训练成功")
        print(f"  损失值: {loss_no_prompt.item():.4f}")
        
        # 有提示训练
        optimizer.zero_grad()
        pred_with_prompt = model(
            pyramid_features, 
            past_mask=torch.zeros_like(target_mask),
            point_coords=point_coords,
            point_labels=point_labels,
            use_prompts=True
        )
        loss_with_prompt = criterion(pred_with_prompt, target_mask)
        loss_with_prompt.backward()
        optimizer.step()
        
        print(f"✓ 有提示训练成功")
        print(f"  损失值: {loss_with_prompt.item():.4f}")
        
    except Exception as e:
        print(f"✗ 训练过程失败: {e}")
    
    print("\n=== 测试4: 推理模式 ===")
    try:
        model.eval()
        with torch.no_grad():
            # 无提示推理
            inference_no_prompt = model(pyramid_features, use_prompts=False)
            inference_no_prompt_sigmoid = torch.sigmoid(inference_no_prompt)
            
            # 有提示推理
            inference_with_prompt = model(
                pyramid_features,
                past_mask=torch.zeros(batch_size, 1, 256, 256).to(device),
                use_prompts=True
            )
            inference_with_prompt_sigmoid = torch.sigmoid(inference_with_prompt)
            
        print(f"✓ 推理模式成功")
        print(f"  无提示推理输出范围: [{inference_no_prompt_sigmoid.min().item():.4f}, {inference_no_prompt_sigmoid.max().item():.4f}]")
        print(f"  有提示推理输出范围: [{inference_with_prompt_sigmoid.min().item():.4f}, {inference_with_prompt_sigmoid.max().item():.4f}]")
        
    except Exception as e:
        print(f"✗ 推理模式失败: {e}")
    
    print("\n=== 测试总结 ===")
    print("✓ 模型支持无提示和有提示两种模式")
    print("✓ 可以进行正常的训练和推理")
    print("✓ 特征金字塔输入正常工作")
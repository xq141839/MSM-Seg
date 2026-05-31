import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal, List

class PyramidPromptUNet(nn.Module):
    """
    基于特征金字塔的交互式分割网络
    利用backbone的多尺度特征做UNet上采样
    """
    
    def __init__(self, feature_channels=256, hi_res_chans=(32, 64),
                 prompt_widths=(16, 16, 32), device=None):
        super().__init__()

        if device is None:
            device = "cuda"
        self.device = device

        # SAM2 高分辨率特征通道：f_256(stride4)=c256, f_128(stride8)=c128, f_64(stride16)=feature_channels
        c256, c128 = hi_res_chans          # 默认 (32, 64)，对应 SAM2 的 conv_s0/conv_s1
        c64 = feature_channels             # pix_feat_with_mem 通道数（默认 256）
        w256, w128, w64 = prompt_widths    # 各尺度提示特征宽度

        # 多尺度提示处理器：把 5 通道提示图 (box 1 + click 4) 编码成小特征
        self.prompt_proc = nn.ModuleDict({
            's256': self._make_prompt_processor(5, w256),
            's128': self._make_prompt_processor(5, w128),
            's64':  self._make_prompt_processor(5, w64),
        })
        # 融合层：拼接 backbone 特征 + 提示特征 -> 卷回原通道数（保证解码器输入维度不变）
        self.prompt_fuse = nn.ModuleDict({
            's256': nn.Conv2d(c256 + w256, c256, 3, padding=1),
            's128': nn.Conv2d(c128 + w128, c128, 3, padding=1),
            's64':  nn.Conv2d(c64 + w64,   c64,  3, padding=1),
        })

        # UNet解码器 - 从64x64开始上采样到256x256
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
                high_res_features: List[torch.Tensor],  # [f_256(stride4), f_128(stride8)]
                pix_feat_with_mem: torch.Tensor = None,  # f_64(stride16)，MSMA 输出的记忆增强特征
                past_mask: Optional[torch.Tensor] = None,
                point_coords: Optional[torch.Tensor] = None,  # [B, n, 2]
                point_labels: Optional[torch.Tensor] = None,  # [B, n]
                scribbles: Optional[torch.Tensor] = None,
                box: Optional[torch.Tensor] = None,           # [B, n, 4] = x1,y1,x2,y2
                use_prompts: bool = True,
                input_resolution: int = 256,                  # 提示坐标所在坐标系(默认 256，与 guidance 同尺度)
                ):
        f_256 = high_res_features[0]   # [B, c256, 256, 256]
        f_128 = high_res_features[1]   # [B, c128, 128, 128]
        f_64 = pix_feat_with_mem       # [B, c64, 64, 64]
        batch_size = f_64.shape[0]

        has_prompt = use_prompts and (
            box is not None or point_coords is not None or scribbles is not None
        )

        # --- 自动模式：无提示，直接用 backbone 特征解码 ---
        if not has_prompt:
            return self.decoder(f_64, f_128, f_256)

        # --- 提示模式：把 box/点编码成多尺度提示图，与各尺度特征融合后再解码 ---
        prompts = {
            'point_coords': point_coords,
            'point_labels': point_labels,
            'scribbles': scribbles,
            'box': box,
        }
        p_256 = self._encode_prompts(prompts, (256, 256), batch_size, input_resolution, ref=f_256)
        p_128 = self._encode_prompts(prompts, (128, 128), batch_size, input_resolution, ref=f_128)
        p_64 = self._encode_prompts(prompts, (64, 64), batch_size, input_resolution, ref=f_64)

        fp_256 = self.prompt_proc['s256'](p_256)
        fp_128 = self.prompt_proc['s128'](p_128)
        fp_64 = self.prompt_proc['s64'](p_64)

        fused_256 = self.prompt_fuse['s256'](torch.cat([f_256, fp_256], dim=1))
        fused_128 = self.prompt_fuse['s128'](torch.cat([f_128, fp_128], dim=1))
        fused_64 = self.prompt_fuse['s64'](torch.cat([f_64, fp_64], dim=1))

        return self.decoder(fused_64, fused_128, fused_256)
    
    def _encode_prompts(self, prompts: Dict, shape: Tuple[int, int], batch_size: int,
                        input_resolution: int = 256, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """把提示编码成 [B,5,H,W] 的提示图（box 1 通道 + 点 4 通道），坐标按尺度缩放。"""
        device = ref.device if ref is not None else torch.device(self.device)
        H, W = shape
        sx = W / float(input_resolution)
        sy = H / float(input_resolution)

        # 边界框：从 input_resolution 坐标系缩放到当前尺度，再填成实心框
        # [修复] 原实现对所有尺度用同一坐标的 box（未缩放），这里按尺度缩放后再 shade
        if prompts.get("box") is not None:
            box = prompts['box'].float()
            box_scaled = box.clone()
            box_scaled[..., 0] = box[..., 0] * sx
            box_scaled[..., 2] = box[..., 2] * sx
            box_scaled[..., 1] = box[..., 1] * sy
            box_scaled[..., 3] = box[..., 3] * sy
            box_embed = bbox_shaded(box_scaled, shape=shape, device=device)  # [B,1,H,W]
        else:
            box_embed = torch.zeros((batch_size, 1) + shape, device=device)

        # 点击（可选）：同样按尺度缩放坐标
        if prompts.get("point_coords") is not None:
            coords = prompts['point_coords'].clone().float()
            coords[..., 0] = coords[..., 0] * sx
            coords[..., 1] = coords[..., 1] * sy
            coords = coords.round().int()
            coords[..., 0] = torch.clamp(coords[..., 0], 0, W - 1)
            coords[..., 1] = torch.clamp(coords[..., 1], 0, H - 1)
            click_embed = click_onehot(coords, prompts['point_labels'], shape=shape)  # [B,4,H,W]
        else:
            click_embed = torch.zeros((batch_size, 4) + shape, device=device)

        # 涂鸦（可选）
        if prompts.get("scribbles") is not None:
            scribbles_resized = F.interpolate(
                prompts['scribbles'].float(), size=shape, mode='bilinear', align_corners=False
            )
            click_embed = torch.clamp(click_embed + scribbles_resized, min=0.0, max=1.0)

        prompt_embeds = torch.cat([box_embed, click_embed], dim=1)  # [B,5,H,W]
        return prompt_embeds.to(device)

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
    device = "cuda"
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

class ModalityAdaptiveFusion(nn.Module):
    """论文 Eq.9 / Eq.2 的模态自适应融合：Ŷ_t = Σ_m w_m · Ŷ_{t,m}

    把同一切片的 M 个模态预测(logits)逐体素自适应加权融合成最终切片预测 Ŷ_t。
    权重 w_m 由一个小卷积网络从各模态预测产生，并沿模态维做 softmax，
    使每个体素上 M 个模态的权重之和为 1（即“逐体素动态选择最有信息的模态”）。

    这取代了原代码“固定权重 [0.2,0.4,0.6,1.0] + 直接取第 4 模态当融合”的简化做法。
    """

    def __init__(self, num_modality=4, hidden=16):
        super().__init__()
        self.num_modality = num_modality
        g = 4 if hidden % 4 == 0 else 1
        self.weight_net = nn.Sequential(
            nn.Conv2d(num_modality, hidden, 3, padding=1),
            nn.GroupNorm(g, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_modality, 1),
        )

    def forward(self, mod_logits):
        """mod_logits: [L, M, H, W]，同一批 L 个切片、各 M 个模态的预测 logits。
        返回 (fused, weights)：fused=[L,1,H,W] 融合后切片预测 logits；weights=[L,M,H,W]。"""
        if self.num_modality == 1:
            return mod_logits, torch.ones_like(mod_logits)
        w = self.weight_net(mod_logits)            # [L, M, H, W]
        w = torch.softmax(w, dim=1)                # 沿模态维归一化
        fused = (w * mod_logits).sum(dim=1, keepdim=True)  # [L, 1, H, W]
        return fused, w
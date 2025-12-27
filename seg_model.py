import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, List

from linear_head import LinearHead


class FrozenDINOv3Backbone(nn.Module):
    """Wraps a frozen DINOv3 model and outputs spatial patch features.

    Input expected shape: (B,3,H,W) already resized & normalized.
    Output: (B, C, H/16, W/16) spatial tensor of patch tokens.
    """
    def __init__(self, model_name: str = 'facebook/dinov3-vith16plus-pretrain-lvd1689m', num_feature_levels: int = 4):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch_size = 16
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', 32)
        self.num_feature_levels = max(1, min(num_feature_levels, self.num_layers))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Return a list of spatial feature maps from multiple transformer layers.
        with torch.no_grad():
            out = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple(len = 1 + num_layers)
        # Choose indices evenly spaced among transformer layers (exclude embedding at index 0).
        L = self.num_layers
        levels = self.num_feature_levels
        stride = max(1, L // levels)
        # hidden_states indexing: last layer at -1, first block at 1.
        # We'll pick from the end backwards with stride.
        idxs = []
        cur = len(hidden_states) - 1
        for _ in range(levels):
            idxs.append(cur)
            cur = max(1, cur - stride)
        # Keep order from shallow to deep for consistency
        idxs = sorted(set(idxs))

        B = x.shape[0]
        H_in, W_in = x.shape[2], x.shape[3]
        H_p, W_p = H_in // self.patch_size, W_in // self.patch_size
        feats: List[torch.Tensor] = []
        for idx in idxs:
            hs = hidden_states[idx]  # (B, T, C)
            patch_tokens = hs[:, 5:, :]  # remove cls + 4 registers
            N = patch_tokens.shape[1]
            assert H_p * W_p == N, f"Token count mismatch: {H_p}*{W_p} != {N} at layer idx={idx}"
            spatial = patch_tokens.view(B, H_p, W_p, self.hidden_size).permute(0, 3, 1, 2).contiguous()
            feats.append(spatial)
        return feats  # list[(B,C,H_p,W_p), ...]


class TAMGatedFusion(nn.Module):
    """
    Fuses TAM features into DINOv3 features using a Gated Mechanism.
    
    For each DINOv3 feature map F_i:
      1. Encode TAM to match DINOv3 channels: TAM' = Conv(TAM)
      2. Compute Gate: G = Sigmoid(Conv(Concat(F_i, TAM')))
      3. Fuse: F_out = F_i + alpha * (G * TAM')
    """
    def __init__(self, dino_channels: int, tam_channels: int, fusion_dim: int = 128):
        super().__init__()
        # We project TAM to dino_channels directly to allow element-wise addition/modulation
        self.tam_encoder = nn.Sequential(
            nn.Conv2d(tam_channels, fusion_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, dino_channels, kernel_size=1)
        )
        
        # Gate generator: takes concatenated [DINO, TAM_encoded] -> 1 channel gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dino_channels * 2, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable scaling factor for the TAM contribution
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, dino_feats: List[torch.Tensor], tam: torch.Tensor) -> List[torch.Tensor]:
        # tam: (B, tam_channels, H, W) - assumed to be resized to match dino_feats spatial dim
        
        # Encode TAM once (assuming all dino feats have same spatial size, which they do in DINOv3 patch output)
        # If dino feats have different sizes, we'd need to resize TAM for each.
        # In FrozenDINOv3Backbone, all outputs are (H/16, W/16).
        
        tam_encoded = self.tam_encoder(tam) # (B, C_dino, H, W)
        
        out_feats = []
        for feat in dino_feats:
            # feat: (B, C_dino, H, W)
            
            # Compute Gate
            cat_feat = torch.cat([feat, tam_encoded], dim=1)
            gate = self.gate_conv(cat_feat) # (B, 1, H, W)
            
            # Modulate TAM features with gate
            gated_tam = gate * tam_encoded
            
            # Fuse: DINO + alpha * GatedTAM
            fused = feat + self.alpha * gated_tam
            out_feats.append(fused)
            
        return out_feats


class SegmentationModel(nn.Module):
    """DINOv3 frozen backbone + (optional) TAM maps fused via Gated Mechanism + LinearHead decoder."""
    def __init__(
        self,
        dino_model_name: str = 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        num_classes: int = 19,
        tam_channels: int = 19,
        use_tam: bool = True,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
    ):
        super().__init__()
        self.backbone = FrozenDINOv3Backbone(dino_model_name, num_feature_levels=num_feature_levels)
        self.use_tam = use_tam
        
        # LinearHead expects a list of feature maps; specify per-level channels.
        # With Gated Fusion, the output channels remain same as backbone hidden size.
        in_channels = [self.backbone.hidden_size for _ in range(self.backbone.num_feature_levels)]
        self.tam_channels = tam_channels
        
        if self.use_tam:
            # Initialize Gated Fusion Module
            self.tam_fusion = TAMGatedFusion(
                dino_channels=self.backbone.hidden_size,
                tam_channels=tam_channels,
                fusion_dim=128 # Intermediate dim for efficiency
            )
            # Note: We do NOT append tam_channels to in_channels anymore, 
            # because TAM is fused INTO the backbone features.
            
        self.decoder = LinearHead(
            in_channels=in_channels,
            n_output_channels=num_classes,
            use_batchnorm=True,
            use_cls_token=False,
            dropout=dropout,
        )

    def forward(self, img: torch.Tensor, tam: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats_list = self.backbone(img)  # list of (B,C,H_p,W_p)
        if self.use_tam:
            target_hw = feats_list[0].shape[2:]
            if tam is not None:
                if tam.shape[2:] != target_hw:
                    tam = torch.nn.functional.interpolate(tam, size=target_hw, mode='bilinear', align_corners=False)
            else:
                B = img.shape[0]
                tam = torch.zeros(B, self.tam_channels, target_hw[0], target_hw[1], device=img.device, dtype=feats_list[0].dtype)
            
            # Apply Gated Fusion
            feats_list = self.tam_fusion(feats_list, tam)
            
        logits = self.decoder(feats_list)  # (B,num_classes,H_p,W_p)
        return logits

    def predict(self, img: torch.Tensor, tam: Optional[torch.Tensor] = None, out_size=(512,1024)) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feats_list = self.backbone(img)
            if self.use_tam:
                target_hw = feats_list[0].shape[2:]
                if tam is not None:
                    if tam.shape[2:] != target_hw:
                        tam = torch.nn.functional.interpolate(tam, size=target_hw, mode='bilinear', align_corners=False)
                else:
                    B = img.shape[0]
                    tam = torch.zeros(B, self.tam_channels, target_hw[0], target_hw[1], device=img.device, dtype=feats_list[0].dtype)
                
                # Apply Gated Fusion
                feats_list = self.tam_fusion(feats_list, tam)
                
            logits = self.decoder.predict(feats_list, rescale_to=out_size)
            return logits

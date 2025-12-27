"""Quick forward shape and gradient check for the segmentation model."""
import torch
from seg_model import SegmentationModel


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegmentationModel(num_feature_levels=3).to(device)
    model.train()
    dummy_img = torch.randn(2,3,512,1024, device=device)
    dummy_tam = torch.randn(2,19,32,64, device=device)
    logits = model(dummy_img, dummy_tam)
    print('Logits shape (patch grid):', logits.shape)
    up = torch.nn.functional.interpolate(logits, size=dummy_img.shape[2:], mode='bilinear')
    print('Upsampled logits shape:', up.shape)
    loss = torch.nn.functional.cross_entropy(up, torch.zeros(2,512,1024, dtype=torch.long, device=device), ignore_index=255)
    loss.backward()
    print('Backward pass OK')


if __name__ == '__main__':
    main()

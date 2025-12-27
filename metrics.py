import torch


def compute_confusion(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """Compute confusion matrix components for mIoU.

    pred: (B,H,W) int64 predicted trainIds
    target: (B,H,W) int64 ground truth trainIds
    Returns: intersections (num_classes,), unions (num_classes,)
    """
    assert pred.shape == target.shape
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    intersections = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum()
        union = (p | t).sum()
        intersections[cls] = inter
        unions[cls] = union
    return intersections, unions


def compute_miou(intersections: torch.Tensor, unions: torch.Tensor):
    ious = []
    for i in range(len(intersections)):
        if unions[i] > 0:
            ious.append((intersections[i] / unions[i]).item())
    if len(ious) == 0:
        return 0.0, []
    return sum(ious) / len(ious), ious

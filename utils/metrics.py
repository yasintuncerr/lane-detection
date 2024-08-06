import numpy as np
from sklearn.metrics import jaccard_score

# Mean IoU calculation function
def mean_iou(preds, labels, num_classes):
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    if preds_flat.shape[0] != labels_flat.shape[0]:
        raise ValueError(f"Predictions and labels have mismatched shapes: "
                         f"{preds_flat.shape} vs {labels_flat.shape}")

    iou = jaccard_score(labels_flat.cpu().numpy(), preds_flat.cpu().numpy(),
                        average=None, labels=range(num_classes))

    return np.mean(iou)

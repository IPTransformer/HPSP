# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

import cv2

def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    # if numerator[0,0] is torch.tensor('inf'):
    #     print(inputs.shape, numerator)

    if torch.any(torch.isnan(numerator)):
        import pdb; pdb.set_trace()

    denominator = inputs.sum(-1, dtype=inputs.dtype)[:, None] + targets.sum(-1, dtype=inputs.dtype)[None, :]
    loss = 1 - (numerator + 1) / (denominator.to(inputs.dtype) + 1)

    # if torch.any(torch.isnan(denominator)):
    #     import pdb; pdb.set_trace()

    # print(inputs.dtype, targets.dtype, numerator.dtype, denominator.dtype, loss.dtype, inputs.sum(-1, )[:, None].dtype)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos.to(inputs.dtype), targets) + torch.einsum(
        "nc,mc->nm", focal_neg.to(inputs.dtype), (1 - targets)
    )

    # if torch.any(torch.isnan(loss)):
    #     import pdb; pdb.set_trace()


    return loss.float() / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        # bs, num_humans = outputs["pred_logits"].shape[:2]
        # outputs = outputs.to(torch.float64)
        # targets = targets.to(torch.float64)

        bs, num_humans = outputs.shape[:2]

        # Work out the mask padding size

        indices = []

        # Iterate through batch size
        for b in range(bs):



            # out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_humans, num_classes]
            out_mask = outputs[b]  # [num_humans, H_pred, W_pred]


            tgt_mask = targets[b].to(out_mask)
            tgt_ids = torch.sum(tgt_mask.sum(dim=(1,2))>0)
            tgt_mask = tgt_mask[:tgt_ids]
            # print(tgt_mask.shape, out_mask.shape)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]

            # Downsample gt masks to save memory
            # tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")
            tgt_mask = tgt_mask.unsqueeze(dim=1)
            # print(tgt_mask.shape, out_mask.shape)

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_humans, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)
            # cost_dice = dice_coeff(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                # + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_humans, -1).cpu()
            # print(C, cost_dice, cost_mask)

            # C = np.asarray(C)

            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_humans, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_humans, H_pred, W_pred] with the predicted masks
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_humans, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

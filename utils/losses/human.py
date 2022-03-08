import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss



def dice_loss(inputs, targets, num_masks):
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
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
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
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # if loss.mean(dim=(1,2)).sum()<0:
    #     print(prob.min(), p_t.min(), prob.max(), targets.max())
    return loss.mean(dim=(1,2)).sum() / num_masks



class HumanLoss(_Loss):
    def __init__(self):
        super(HumanLoss, self).__init__()


    def forward(self, preds, targets):

        N = targets.shape[0]
        # print(preds.shape, targets.shape)
        # print('1: ', targets.max().item())

        loss_dice = dice_loss(inputs=preds, targets=targets, num_masks=N)
        focal_loss = sigmoid_focal_loss(preds, targets, N)

        # print(focal_loss, loss_dice)



        return loss_dice + 20*focal_loss

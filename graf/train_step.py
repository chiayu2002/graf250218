import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import torch.nn as nn




def compute_loss(d_outs, target):

        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = 0

        for d_out in d_outs:
            targets = d_out.new_full(size=d_out.size(), fill_value=target)
            loss += F.binary_cross_entropy_with_logits(d_out, targets)
        return loss / len(d_outs)


def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

# def compute_cls_loss(P_prime_cls, I_cls, class_labels):
#         """
#         計算分類損失 Lcls
#         Args:
#             P_prime: 生成的patch
#             I: 原始圖像
#             class_labels: 圖像的類別標籤
#         Returns:
#             cls_loss: 分類損失
#         """
#         device = I_cls.device
#         class_labels = class_labels.to(device)
#         class_labels = class_labels[:, 0].long()
#         # 計算生成patch和原始圖像的分類損失
#         P_prime_loss = nn.functional.cross_entropy(P_prime_cls, class_labels)
#         I_loss = nn.functional.cross_entropy(I_cls, class_labels)
        
#         # 總分類損失
#         total_cls_loss = P_prime_loss + I_loss
        
#         return total_cls_loss

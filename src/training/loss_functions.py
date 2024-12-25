import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_label_loss(student_logits, labels):
   ce_loss = F.cross_entropy(student_logits, labels)
   return ce_loss

def calculate_representation_loss(teacher_hidden, student_hidden):
    # Representation alignment loss (MSE)
    return F.mse_loss(student_hidden, teacher_hidden)

def calculate_logit_distill_loss(student_logits, teacher_logits, temperature):
    # Logit alignment loss (KL-Divergence)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

'''
def combined_label_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Cross-Entropy Loss (Student vs Ground Truth)
    label_loss =  calculate_label_loss(student_logits, labels)

    # Distillation Loss (Student vs Teacher)
    distill_loss =  calculate_logit_distill_loss(student_logits, teacher_logits, temperature)

    # Combined Loss: Weighted sum of distillation and cross-entropy losses
    total_loss = alpha * distill_loss + (1 - alpha) * label_loss
    return total_loss
'''


class CombinedLabelLoss(nn.Module):
    """
    Custom loss function for combining cross-entropy loss with distillation loss.

    Args:
    - temperature (float): Temperature for distillation loss.
    - alpha (float): Weighting factor for distillation loss (0 <= alpha <= 1).
    """
    def __init__(self, temperature: float, alpha: float):
        super(CombinedLabelLoss, self).__init__()
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, labels, student_logits, teacher_logits, student_hidden, teacher_hidden):
        """
        Compute the combined loss.

        Args:
        - student_logits (torch.Tensor): Logits output by the student model. Shape: [batch_size, num_classes]
        - teacher_logits (torch.Tensor): Logits output by the teacher model. Shape: [batch_size, num_classes]
        - labels (torch.Tensor): Ground truth labels. Shape: [batch_size]
        
        Returns:
        - total_loss (torch.Tensor): Combined loss value.
        """
        label_loss = calculate_label_loss(student_logits, labels)

        distill_loss = calculate_logit_distill_loss(student_logits, teacher_logits, self.temperature)

        total_loss = self.alpha * distill_loss + (1 - self.alpha) * label_loss
        return total_loss

class CombinedReprLoss(nn.Module):
    """
    Custom loss function for combining cross-entropy loss with distillation loss.

    Args:
    - temperature (float): Temperature for distillation loss.
    - alpha (float): Weighting factor for distillation loss (0 <= alpha <= 1).
    """
    def __init__(self, alpha, temperature):
        super(CombinedLabelLoss, self).__init__()
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= beta <= 1, "Beta must be between 0 and 1"
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, labels, student_logits, teacher_logits, student_hidden, teacher_hidden):
        label_loss = calculate_label_loss(student_logits, labels)

        rep_loss = calculate_representation_loss(teacher_hidden, student_hidden)

        distill_loss = calculate_logit_distill_loss(student_logits, teacher_logits, self.temperature)

        total_loss = self.alpha * rep_loss + self.beta * distill_loss + (1 - self.alpha - self.beta) * label_loss
        return total_loss

'''
def combined_repr_loss(student_logits, teacher_logits, teacher_hidden, student_hidden, labels, temperature, alpha, beta):
    # Cross-Entropy Loss with ground truth labels
    label_loss =  calculate_label_loss(student_logits, labels)

    # Representation Distillation Loss
    rep_loss = calculate_representation_loss(teacher_hidden, student_hidden)

    # Logit Distillation Loss (optional)
    distill_loss = calculate_logit_distill_loss(student_logits, teacher_logits, temperature)

    # Total Loss
    total_loss = alpha * rep_loss + beta * distill_loss + (1 - alpha - beta) * label_loss
    return total_loss
'''



def get_loss_class(loss_type, alpha, beta, temperature):
  if loss_type == "logits":
     loss_obj = CombinedLabelLoss(alpha, temperature)
  elif loss_type == "CLS":
     loss_obj = CombinedReprLoss(alpha, beta, temperature)
  return loss_obj

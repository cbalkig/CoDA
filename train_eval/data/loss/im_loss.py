import torch
import torch.nn as nn
import torch.nn.functional as F


class IMLoss(nn.Module):
    def __init__(self, num_classes, diversity_weight=1.0, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.diversity_weight = diversity_weight
        self.epsilon = epsilon

    def forward(self, logits):
        """
        Args:
            logits: (Batch_Size, Num_Classes)
        """
        # 1. Get Probabilities (Softmax)
        probs = F.softmax(logits, dim=1)

        # 2. Conditional Entropy (H(Y|X)) - "Be Confident"
        # Measure uncertainty of each sample individually
        # Formula: - sum(p * log(p)) per sample, then average over batch
        entropy_per_sample = -torch.sum(probs * torch.log(probs + self.epsilon), dim=1)
        conditional_entropy = torch.mean(entropy_per_sample)

        # 3. Marginal Entropy (H(Y)) - "Be Diverse"
        # Measure uncertainty of the AVERAGE prediction across the batch
        # Formula: Average the probs first, then calc entropy
        avg_probs = torch.mean(probs, dim=0)  # Shape: (Num_Classes,)
        marginal_entropy = -torch.sum(avg_probs * torch.log(avg_probs + self.epsilon))

        # 4. Total IM Loss
        # We want to MINIMIZE Conditional Entropy (Confidence)
        # We want to MAXIMIZE Marginal Entropy (Diversity) -> so we subtract it
        loss = conditional_entropy - (self.diversity_weight * marginal_entropy)

        return loss, conditional_entropy, marginal_entropy

"""Current-lap pit classifier. Inputs are decision-time features in FEATURES order."""

import torch
from torch import nn
from torch.nn import functional as F


FEATURES = ('race_progress', 'position_fraction', 'previous_pits')


class PitMLP(nn.Module):
    """Affine -> ReLU -> affine. The default 3 -> 8 -> 1 model has 41 parameters.

    forward(X) returns one logit per row. loss(X, y) returns mean binary
    cross-entropy; loss.backward() populates the parameter gradients.
    The caller owns the optimizer, race splits, and training loop.
    """

    def __init__(self, input_dim=len(FEATURES), hidden_dim=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold = None  # Set on validation data after fitting the weights.

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    def _labeled_batch(self, x, y):
        scores = self(x)
        targets = torch.as_tensor(y, device=scores.device)
        if scores.ndim != 1 or scores.shape != targets.shape or scores.numel() == 0:
            raise ValueError("Expected a nonempty batch X (N, D) and y (N,).")
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("Pit labels must be 0 or 1.")
        if not scores.isfinite().all():
            raise ValueError("Nonfinite logits. Check the features and weights.")
        return scores, targets.to(scores.dtype)

    def loss(self, x, y):
        """Training objective. y[i] is 1 when the driver pits during lap i's example."""
        scores, targets = self._labeled_batch(x, y)
        return F.binary_cross_entropy_with_logits(scores, targets)

    @torch.no_grad()
    def predict_proba(self, x):
        return self(x).sigmoid()

    def _decision_threshold(self):
        if self.threshold is None:
            raise ValueError("Select a threshold on a labeled validation set first.")
        return self.threshold

    @torch.no_grad()
    def predict(self, x):
        return self.predict_proba(x) >= self._decision_threshold()

    @staticmethod
    def _precision_recall_curve(probabilities, targets):
        # Each distinct score is one possible threshold. Include all tied
        # observations together, exactly as predict() does with >=.
        ordered, indices = probabilities.sort(descending=True, stable=True)
        levels, counts = torch.unique_consecutive(ordered, return_counts=True)
        predicted = counts.cumsum(0)
        positives = targets[indices].double().cumsum(0)[predicted - 1]
        precision = positives / predicted
        recall = positives / targets.double().sum().clamp_min(1)
        return levels, precision, recall

    @torch.no_grad()
    def select_threshold(self, x_val, y_val):
        """Maximize validation F1; break ties in favor of the highest threshold."""
        scores, targets = self._labeled_batch(x_val, y_val)
        if targets.min() == targets.max():
            raise ValueError("Validation needs both pit and stay-out examples.")
        levels, precision, recall = self._precision_recall_curve(scores.sigmoid(), targets)
        total = precision + recall
        f1 = torch.where(total > 0, 2 * precision * recall / total, 0.)
        self.threshold = levels[f1.argmax()].item()
        return self.threshold

    @torch.no_grad()
    def evaluate(self, x, y):
        """Use the stored threshold. Undefined precision, recall or AP is reported as zero."""
        threshold = self._decision_threshold()
        scores, targets = self._labeled_batch(x, y)
        probabilities = scores.sigmoid()
        predicted = probabilities >= threshold
        positives = targets.bool()
        tp = (predicted & positives).sum().item()
        n_pred, n_pos = predicted.sum().item(), positives.sum().item()
        _, precision, recall = self._precision_recall_curve(probabilities, targets)
        average_precision = (recall.diff(prepend=recall.new_zeros(1)) * precision).sum()
        return {
            'bce': F.binary_cross_entropy_with_logits(scores, targets).item(),
            'precision': tp / max(n_pred, 1),
            'recall': tp / max(n_pos, 1),
            'f1': 2 * tp / max(n_pred + n_pos, 1),
            'ap': average_precision.item(),
        }

    def save_checkpoint(self, path):
        """Store architecture, feature order, weights and validated decision threshold."""
        if self.layers[0].in_features != len(FEATURES):
            raise ValueError("Project checkpoints require the three project features.")
        torch.save({
            'features': FEATURES,
            'hidden_dim': self.layers[0].out_features,
            'state_dict': self.state_dict(),
            'threshold': self._decision_threshold(),
        }, path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if tuple(checkpoint['features']) != FEATURES:
            raise ValueError("Checkpoint feature order differs from the project schema.")
        threshold = float(checkpoint['threshold'])
        if not 0 <= threshold <= 1:
            raise ValueError("Invalid decision threshold in checkpoint.")
        weights = checkpoint['state_dict']
        model = cls(hidden_dim=checkpoint['hidden_dim']).to(weights['layers.0.weight'])
        model.load_state_dict(weights)
        model.threshold = threshold
        return model.eval()

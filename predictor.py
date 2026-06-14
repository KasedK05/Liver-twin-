import torch
import numpy as np
from pathlib import Path
import json

class Predictor:
    """Run inference on raw test data using trained model."""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)

        from models.hierarchical_model import HierarchicalCellModel
        self.model = HierarchicalCellModel(
            input_dim=32,
            architecture=checkpoint['architecture'],
            num_overall_classes=6,
            num_mechanism_classes=len(checkpoint['mech_to_idx'])
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.mech_to_idx = checkpoint['mech_to_idx']
        self.idx_to_mech = {v: k for k, v in self.mech_to_idx.items()}
        self.overall_classes = ["CVR", "BRT", "PVR", "VBR", "TRF", "SMR"]

    def predict(self, trajectories, times=None):
        """Predict on raw test trajectories."""
        if times is None:
            times = np.arange(49) * 0.5

        results = []
        with torch.no_grad():
            for i in range(len(trajectories)):
                x = torch.from_numpy(trajectories[i]).unsqueeze(0).float().to(self.device)
                t = torch.from_numpy(times).unsqueeze(0).float().to(self.device)

                overall_logits, mech_logits, sub_preds, attn = self.model(x, t)

                overall_prob = torch.softmax(overall_logits, dim=1).cpu().numpy()[0]
                mech_prob = torch.softmax(mech_logits, dim=1).cpu().numpy()[0]
                sub_health = sub_preds.cpu().numpy()[0]
                attn_weights = attn.cpu().numpy()[0]

                overall_pred = self.overall_classes[overall_prob.argmax()]
                mech_pred = self.idx_to_mech[mech_prob.argmax()]

                results.append({
                    'overall': overall_pred,
                    'overall_confidence': float(overall_prob.max()),
                    'overall_probs': {cls: float(p) for cls, p in zip(self.overall_classes, overall_prob)},
                    'mechanism': mech_pred,
                    'mechanism_confidence': float(mech_prob.max()),
                    'subsystem_health': {
                        'genetic': float(sub_health[0]),
                        'energy': float(sub_health[1]),
                        'protein': float(sub_health[2]),
                        'signaling': float(sub_health[3]),
                        'membrane': float(sub_health[4]),
                        'defense': float(sub_health[5]),
                        'waste': float(sub_health[6]),
                    },
                    'attention_weights': attn_weights.tolist()
                })

        return results

    def evaluate(self, trajectories, labels, mechanisms=None):
        """Evaluate model on test set."""
        predictions = self.predict(trajectories)

        pred_labels = [p['overall'] for p in predictions]
        true_labels = list(labels)

        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels, labels=self.overall_classes).tolist()
        }

        return metrics, predictions

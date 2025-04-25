import torch
from torch.utils import data
import random

class TwitterDataset(data.Dataset):
    """
    Dataset for handling both positive-negative samples and single sample data.
    The dataset can be used in two modes:
    1. Positive-Negative mode: Returns anchor, positive, and negative samples (default).
    2. Single sample mode: Returns only anchor samples (if `mode='single'` is passed).
    """

    def __init__(self, x, z, mode='positive_negative'):
        """
        Args:
            x: Data (e.g., gene expression or other features), shape: (data_num, seq_len, feature_dim)
            z: Labels, shape: (data_num, label_dim)
            mode: 'positive_negative' for both positive and negative samples,
                  'single' for single sample data (defaults to 'positive_negative')
        """
        self.data1 = x
        self.label = z
        self.mode = mode

    def __getitem__(self, index):
        anchor_label = self.label[index]
        anchor_gene = self.data1[index]

        # Single sample mode: return only the anchor
        if self.mode == 'single':
            return anchor_label, anchor_gene

        # Positive-Negative mode: find positive and negative samples
        indices_with_same_label = [i for i, label in enumerate(self.label) if torch.all(label == anchor_label)]
        
        # Add the condition for only one data point with the given label
        if len(indices_with_same_label) == 1:
            positive_index = indices_with_same_label[0]
        else:
            positive_index = random.choice([i for i in indices_with_same_label if i != index])

        positive_label = self.label[positive_index]
        positive_gene = self.data1[positive_index]
        
        # Find negative sample (different label)
        negative_index = random.choice([i for i, label in enumerate(self.label) if torch.any(label != anchor_label)])
        negative_label = self.label[negative_index]
        negative_gene = self.data1[negative_index]

        # Return anchor, positive, and negative samples
        return anchor_label, anchor_gene, positive_label, positive_gene, negative_label, negative_gene

    def __len__(self):
        return len(self.data1)

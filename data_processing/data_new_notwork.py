import torch
from torch.utils import data
import random
from collections import defaultdict

class TwitterDataset(data.Dataset):
    """
    Optimized Dataset for handling both positive-negative samples and single sample data.
    Includes label balancing during initialization and pre-caches positive/negative sample indices.
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
        
        if mode == 'positive_negative':
            # Group indices by label
            self.label_indices = defaultdict(list)
            for idx, lbl in enumerate(z):
                self.label_indices[tuple(lbl.tolist())].append(idx)
            
            # Calculate label counts for balancing
            self.label_counts = {label: len(indices) for label, indices in self.label_indices.items()}
            self.target_count = max(self.label_counts.values())  # Use max count as target count
            
            # Oversample indices for each label
            self.oversampled_indices = []
            for label, indices in self.label_indices.items():
                count = self.label_counts[label]
                if count < 100:  # Only oversample for classes with fewer than 100 samples
                    self.oversampled_indices.extend(indices * (100 // count))
                    self.oversampled_indices.extend(random.choices(indices, k=100 % count))
                else:
                    self.oversampled_indices.extend(indices)
            
            # Pre-cache positive and negative indices for each anchor
            self.positive_indices_cache = {}
            self.negative_indices_cache = {}

            # Cache indices only once during initialization
            for idx in self.oversampled_indices:
                anchor_label = self.label[idx]
                self.positive_indices_cache[idx] = [
                    i for i in self.oversampled_indices if torch.all(self.label[i] == anchor_label)
                ]
                self.negative_indices_cache[idx] = [
                    i for i in self.oversampled_indices if not torch.all(self.label[i] == anchor_label)
                ]

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.label[index], self.data1[index]

        # Use balanced oversampled indices
        balanced_index = self.oversampled_indices[index]
        anchor_label = self.label[balanced_index]
        anchor_gene = self.data1[balanced_index]

        # Retrieve pre-cached positive and negative indices
        positive_indices = self.positive_indices_cache[balanced_index]
        negative_indices = self.negative_indices_cache[balanced_index]

        # Select positive and negative samples
        positive_index = random.choice(positive_indices)
        positive_label = self.label[positive_index]
        positive_gene = self.data1[positive_index]

        negative_index = random.choice(negative_indices)
        negative_label = self.label[negative_index]
        negative_gene = self.data1[negative_index]

        return anchor_label, anchor_gene, positive_label, positive_gene, negative_label, negative_gene

    def __len__(self):
        if self.mode == 'positive_negative':
            return len(self.oversampled_indices)
        else:
            return len(self.data1)

class TwitterDataset_MulitTrain(data.Dataset):
    def __init__(self, x, y, label_x, label_y):
        self.data1 = x
        self.data2 = y
        self.label_x = label_x
        self.label_y = label_y
        
        # Group dataset 2 indices by label
        self.label_indices = defaultdict(list)
        for idx, lbl in enumerate(label_y):
            self.label_indices[tuple(lbl.tolist())].append(idx)
        
        # Oversample indices to balance labels
        self.oversampled_indices = []
        for label, indices in self.label_indices.items():
            count = len(indices)
            target_count = max(len(indices) for indices in self.label_indices.values())
            if count < target_count:
                self.oversampled_indices.extend(indices * (target_count // count))
                self.oversampled_indices.extend(random.choices(indices, k=target_count % count))
            else:
                self.oversampled_indices.extend(indices)

        # Pre-compute label mapping for oversampled dataset
        self.oversampled_labels = [label_y[idx] for idx in self.oversampled_indices]

    def __getitem__(self, index):
        # Anchor sample from dataset 1
        anchor_label = self.label_x[index]
        anchor_gene = self.data1[index]

        # Pre-filtered positive and negative indices
        positive_indices = [
            i for i, lbl in enumerate(self.oversampled_labels) if torch.all(lbl == anchor_label)
        ]
        negative_indices = [
            i for i, lbl in enumerate(self.oversampled_labels) if torch.any(lbl != anchor_label)
        ]

        if not positive_indices:
            return self.__getitem__((index + 1) % len(self))
        positive_index = random.choice(positive_indices)
        positive_label = self.oversampled_labels[positive_index]
        positive_gene = self.data2[self.oversampled_indices[positive_index]]

        negative_index = random.choice(negative_indices)
        negative_label = self.oversampled_labels[negative_index]
        negative_gene = self.data2[self.oversampled_indices[negative_index]]

        return anchor_label, anchor_gene, positive_label, positive_gene, negative_label, negative_gene

    def __len__(self):
        return len(self.data2)
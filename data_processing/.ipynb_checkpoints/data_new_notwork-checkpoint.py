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
        if len(indices_with_same_label) == 0:
            return self.__getitem__((index + 1) % len(self))  # 跳过当前样本，尝试下一个样本
        elif len(indices_with_same_label) == 1:
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

class TwitterDataset_MulitTrain(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, x, y, label_x, label_y):
        self.data1 = x
        self.data2 = y
        self.label_x = label_x  # 数据集1的标签，为整合的物种或组学
        self.label_y = label_y  # 数据集2的标签，为将要整合到的参考物种或组学
        
    def __getitem__(self, index):
        anchor_label = self.label_x[index]
        anchor_gene = self.data1[index]

        #正样本，有所修改
        
        indices_with_same_label = [i for i, label in enumerate(self.label_y) if torch.all(label == anchor_label)]
        if len(indices_with_same_label) == 0:
            return self.__getitem__((index + 1) % len(self))  # 跳过当前样本，尝试下一个样本
        elif len(indices_with_same_label) == 1:
            positive_index = indices_with_same_label[0]
        else:
            positive_index = random.choice([i for i in indices_with_same_label if i != index])
        
        positive_label = self.label_y[positive_index]
        positive_gene = self.data2[positive_index]

        # 在数据集2中找到不同标签的负样本
        negative_index = random.choice([i for i, label in enumerate(self.label_y) if torch.any(label != anchor_label)])
        negative_label = self.label_y[negative_index]
        negative_gene = self.data2[negative_index]

        #暂时未加上基因表达值
        return anchor_label, anchor_gene, positive_label, positive_gene, negative_label, negative_gene
        
    def __len__(self):
        return len(self.data2)
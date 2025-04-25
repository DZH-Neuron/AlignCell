import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time

class Get_Embedding:
    def __init__(self, model_path, device='cuda'):
        """
        初始化 ModelEvaluator 实例，加载模型并设置计算设备。
        
        Args:
            model_path (str): 预训练模型路径。
            device (str): 模型运行的设备，默认为 'cuda'。
        """
        self.device = device
        self.model = torch.load(model_path)
        self.model = self.model.to(self.device)
    
    def get_embeddings(self, dataset):
        """
        获取数据集的嵌入向量及标签。
        
        Args:
            dataset (iterable): 数据集，包含 anchor_label 和 anchor_gene。
        
        Returns:
            vec_record (list): 嵌入向量列表。
            vec_label (list): 标签列表。
        """
        vec_record = []
        vec_label = []
        start_time = time.time()
        
        # 获取嵌入向量
        for anchor_label, anchor_gene in dataset:
            vec_label.extend(anchor_label.flatten().tolist())
            out = self.model(anchor_gene.unsqueeze(0).to(self.device))
            vec_record.extend(out.flatten().tolist())
        
        end_time = time.time()
        print(f"Embedding extraction time: {end_time - start_time} seconds.")
        
        return np.array(vec_label), np.array(vec_record)
    
    def calculate_cosine_similarity(self, vec_data1, vec_data2):
        """
        计算两个数据集之间的余弦相似度，并返回最相似的索引。
        
        Args:
            vec_data1 (numpy.ndarray): 第一个数据集的嵌入向量。
            vec_data2 (numpy.ndarray): 第二个数据集的嵌入向量。
        
        Returns:
            most_similar_indices (list): 与第二个数据集中的每一行最相似的第一数据集的行索引。
        """
        data_tensor1 = torch.tensor(vec_data1, dtype=torch.float).to(self.device)
        data_tensor2 = torch.tensor(vec_data2, dtype=torch.float).to(self.device)

        most_similar_indices = []
        
        # 计算余弦相似度
        for row2 in data_tensor2:
            similarities = F.cosine_similarity(row2.unsqueeze(0), data_tensor1)
            most_similar_index = torch.argmax(similarities).item()
            most_similar_indices.append(most_similar_index)
        
        return most_similar_indices

    def get_embedding_matrix(self, val_dataset1, val_dataset2):
        """
        获取两个数据集的嵌入向量矩阵。
        
        Args:
            val_dataset1 (iterable): 第一个数据集，包含 anchor_label 和 anchor_gene。
            val_dataset2 (iterable): 第二个数据集，包含 anchor_label 和 anchor_gene。
        
        Returns:
            embedding_matrix1 (numpy.ndarray): 第一个数据集的嵌入矩阵。
            embedding_matrix2 (numpy.ndarray): 第二个数据集的嵌入矩阵。
        """
        # 获取第一个数据集的嵌入向量和标签
        vec_label1, vec_record1 = self.get_embeddings(val_dataset1)
        # 获取第二个数据集的嵌入向量和标签
        vec_label2, vec_record2 = self.get_embeddings(val_dataset2)

        # 重塑数据
        vec_data1 = vec_record1.reshape(len(val_dataset1), -1)
        vec_data2 = vec_record2.reshape(len(val_dataset2), -1)

        return vec_data1, vec_data2

    def evaluate(self, val_dataset1, val_dataset2):
        """
        评估两个数据集之间的标签匹配情况。
        
        Args:
            val_dataset1 (iterable): 第一个数据集，包含 anchor_label 和 anchor_gene。
            val_dataset2 (iterable): 第二个数据集，包含 anchor_label 和 anchor_gene。
        
        Returns:
            acc_df (pd.DataFrame): 包含 'rel_label' 和 'test_label' 的对比 DataFrame。
        """
        # 获取第一个数据集的嵌入向量和标签
        vec_label1, vec_record1 = self.get_embeddings(val_dataset1)
        # 获取第二个数据集的嵌入向量和标签
        vec_label2, vec_record2 = self.get_embeddings(val_dataset2)

        # 重塑数据
        label_data1 = vec_label1.reshape(len(val_dataset1), 2)
        label_data2 = vec_label2.reshape(len(val_dataset2), 2)
        vec_data1 = vec_record1.reshape(len(val_dataset1), -1)
        vec_data2 = vec_record2.reshape(len(val_dataset2), -1)

        # 将标签转换为字符串列表
        list_data1 = ['_'.join(map(str, row)) for row in label_data1]
        list_data2 = ['_'.join(map(str, row)) for row in label_data2]

        # 计算最相似的索引
        most_similar_indices = self.calculate_cosine_similarity(vec_data1, vec_data2)

        # 获取最相似的行名
        row_names = [list_data1[i] for i in most_similar_indices]
        rel_name = list_data2
        
        # 创建 DataFrame 对比结果
        acc_df = pd.DataFrame({'rel_label': rel_name, 'test_label': row_names})

        return acc_df

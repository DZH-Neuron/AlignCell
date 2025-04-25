import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class AlignCell_Emb:
    def __init__(self, model, device=None):
        """
        初始化模型评估对象，加载模型并选择合适的计算设备（CUDA 或 CPU）。
        
        Args:
            model: 模型对象，假设我们使用一个模拟模型（MockModel）。
            device (str or None): 计算设备，可以是 'cuda' 或 'cpu'。默认为 None，会自动检测。
        """
        # 检查 CUDA 是否可用，如果没有传入 device 参数，则默认选择可用的设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = torch.load(model, weights_only=False)  # 使用传入的模型
    
    def get_embeddings(self, dataset, output_attentions=False):
        """
        获取数据集的嵌入向量及标签。

        Args:
            dataset (iterable): 数据集，包含 anchor_label 和 anchor_gene。
            output_attentions (bool): 是否返回注意力权重。

        Returns:
            vec_record (list): 嵌入向量列表。
            vec_label (list): 标签列表。
            attentions (list, optional): 注意力权重列表（如果 output_attentions=True）。
        """
        vec_record = []
        vec_label = []
        attentions = [] if output_attentions else None

        self.model.eval()
        # 获取嵌入向量
        for i, (anchor_label, anchor_gene) in enumerate(tqdm(dataset, desc="Processing Dataset")):
            vec_label.extend(anchor_label.flatten().tolist())

            if output_attentions:
                out, attention_weights = self.model(anchor_gene.unsqueeze(0).to(self.device),output_attentions=True)
                attentions.append([layer_weights.detach().cpu().numpy() for layer_weights in attention_weights])
            else:
                out = self.model(anchor_gene.unsqueeze(0).to(self.device))

            vec_record.extend(out.flatten().tolist())

        if output_attentions:
            return np.array(vec_label), np.array(vec_record), np.array(attentions)
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
        print("Calculating cosine similarity......")
        data_tensor1 = torch.tensor(vec_data1, dtype=torch.float).to(self.device)
        data_tensor2 = torch.tensor(vec_data2, dtype=torch.float).to(self.device)

        most_similar_indices = []
        
        # 计算余弦相似度
        for row2 in data_tensor2:
            similarities = F.cosine_similarity(row2.unsqueeze(0), data_tensor1)
            most_similar_index = torch.argmax(similarities).item()
            most_similar_indices.append(most_similar_index)

        print("Finished Calculate cosine similarity!")
        return most_similar_indices

    def evaluate(self, val_dataset1, val_dataset2):
        """
        评估两个数据集之间的标签匹配情况。
        
        Args:
            val_dataset1 (iterable): 第一个数据集，包含 anchor_label 和 anchor_gene。
            val_dataset2 (iterable): 第二个数据集，包含 anchor_label 和 anchor_gene。
        
        Returns:
            acc_df (pd.DataFrame): 包含 'rel_label' 和 'test_label' 的对比 DataFrame。
        """

        print("Begian evalution......")
        # 获取第一个数据集的嵌入向量和标签
        print("Begain Embedding ref data......")
        vec_label1, vec_record1 = self.get_embeddings(val_dataset1)
        # 获取第二个数据集的嵌入向量和标签
        print("Begain Embedding query data......")
        vec_label2, vec_record2 = self.get_embeddings(val_dataset2)
        print("Finished!")

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

        print("Finished all!")
        return acc_df

    def generate_dataframe(self, vec_record, vec_label, dataset, embedding_size=256):
        """
        根据嵌入向量和标签生成 Pandas DataFrame。
        
        Args:
            vec_record (np.array): 嵌入向量数组。
            vec_label (np.array): 标签数组。
            val_dataset (iterable): 验证数据集，用于确定数据集长度。
            embedding_size (int): 嵌入向量的尺寸。
        
        Returns:
            df (pd.DataFrame): 带索引的嵌入向量 DataFrame。
        """
        # 调整嵌入向量形状
        vec_data = vec_record.reshape(len(dataset), embedding_size)

        # 假设标签是二维的
        label_data = vec_label.reshape(len(dataset), 2)
        index_labels = ['_'.join(map(str, row)) for row in label_data]

        # 创建 DataFrame
        df = pd.DataFrame(vec_data)
        df.index = index_labels
        return df
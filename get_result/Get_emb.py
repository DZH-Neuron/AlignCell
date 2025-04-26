import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class AlignCell_Emb:
    def __init__(self, model, device=None):
        """
        Initialize the model evaluation object, load the model, and select the appropriate computation device (CUDA or CPU).
    
        Args:
            model: Model object, assuming we are using a mock model (MockModel).
            device (str or None): Computation device, can be 'cuda' or 'cpu'. Default is None, which will automatically detect the available device.
        """
        # Check if CUDA is available, and if no device argument is passed, select the available device by default.
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = torch.load(model, weights_only=False) 
    
    def get_embeddings(self, dataset, output_attentions=False):
        """
        Retrieve the embedding vectors and labels of the dataset.

        Args:
            dataset (iterable): The dataset, containing `anchor_label` and `anchor_gene`.
            output_attentions (bool): Whether to return attention weights.

        Returns:
            vec_record (list): A list of embedding vectors.
            vec_label (list): A list of labels.
            attentions (list, optional): A list of attention weights (if `output_attentions=True`).
        """
        vec_record = []
        vec_label = []
        attentions = [] if output_attentions else None

        self.model.eval()
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
        Compute the cosine similarity between two datasets and return the most similar indices.

        Args:
            vec_data1 (numpy.ndarray): Embedding vectors of the first dataset.
            vec_data2 (numpy.ndarray): Embedding vectors of the second dataset.

        Returns:
            most_similar_indices (list): The row indices of the first dataset that are most similar to each row in the second dataset.
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
        Evaluate the label matching between two datasets.

        Args:
            val_dataset1 (iterable): The first dataset, containing `anchor_label` and `anchor_gene`.
            val_dataset2 (iterable): The second dataset, containing `anchor_label` and `anchor_gene`.

        Returns:
            acc_df (pd.DataFrame): A comparison DataFrame containing 'rel_label' and 'test_label'.
        """
        print("Begian evalution......")
        print("Begain Embedding ref data......")
        vec_label1, vec_record1 = self.get_embeddings(val_dataset1)
        print("Begain Embedding query data......")
        vec_label2, vec_record2 = self.get_embeddings(val_dataset2)
        print("Finished!")

        label_data1 = vec_label1.reshape(len(val_dataset1), 2)
        label_data2 = vec_label2.reshape(len(val_dataset2), 2)
        vec_data1 = vec_record1.reshape(len(val_dataset1), -1)
        vec_data2 = vec_record2.reshape(len(val_dataset2), -1)

        list_data1 = ['_'.join(map(str, row)) for row in label_data1]
        list_data2 = ['_'.join(map(str, row)) for row in label_data2]

        most_similar_indices = self.calculate_cosine_similarity(vec_data1, vec_data2)

        row_names = [list_data1[i] for i in most_similar_indices]
        rel_name = list_data2
        
        acc_df = pd.DataFrame({'rel_label': rel_name, 'test_label': row_names})

        print("Finished all!")
        return acc_df

    def generate_dataframe(self, vec_record, vec_label, dataset, embedding_size=256):
        """
        Generate a Pandas DataFrame from embedding vectors and labels.

        Args:
            vec_record (np.array): Array of embedding vectors.
            vec_label (np.array): Array of labels.
            val_dataset (iterable): Validation dataset, used to determine the length of the dataset.
            embedding_size (int): The size of the embedding vectors.

        Returns:
            df (pd.DataFrame): A DataFrame of embedding vectors with indices.
        """
        vec_data = vec_record.reshape(len(dataset), embedding_size)

        label_data = vec_label.reshape(len(dataset), 2)
        index_labels = ['_'.join(map(str, row)) for row in label_data]

        df = pd.DataFrame(vec_data)
        df.index = index_labels
        return df

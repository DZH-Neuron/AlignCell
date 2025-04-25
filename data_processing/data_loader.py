import pandas as pd

class DataLoader:
    def __init__(self, expression_matrix_1, cell_labels_1, expression_matrix_2=None, cell_labels_2=None, top_genes=1000, mode="target", cell_name_column='cell_name', cell_type_column='cell_type'):
        """
        初始化数据加载器，支持处理一个或两个数据集。

        Args:
            expression_matrix_1 (pd.DataFrame): 第一个基因表达矩阵，行是基因名，列是细胞名。
            cell_labels_1 (pd.DataFrame): 第一个细胞标签，包含列用于指定细胞名称和标签。
            expression_matrix_2 (pd.DataFrame, optional): 第二个基因表达矩阵，行是基因名，列是细胞名。默认为 None。
            cell_labels_2 (pd.DataFrame, optional): 第二个细胞标签，包含列用于指定细胞名称和标签。默认为 None。
            top_genes (int): 每个细胞提取的高表达基因数目。
            mode (str): 当前数据模式，支持 'target', 'query', 或 'train'。
            cell_name_column (str): 用于匹配细胞名称的列名，默认是 'cell_name'。
            cell_type_column (str): 用于匹配细胞类型的列名，默认是 'cell_type'。
        """
        # 检查输入数据格式
        if not isinstance(expression_matrix_1, pd.DataFrame):
            raise TypeError("expression_matrix_1 should be a pandas DataFrame.")
        if not isinstance(cell_labels_1, pd.DataFrame):
            raise TypeError("cell_labels_1 should be a pandas DataFrame.")
        if expression_matrix_2 is not None and not isinstance(expression_matrix_2, pd.DataFrame):
            raise TypeError("expression_matrix_2 should be a pandas DataFrame if provided.")
        if cell_labels_2 is not None and not isinstance(cell_labels_2, pd.DataFrame):
            raise TypeError("cell_labels_2 should be a pandas DataFrame if provided.")
        
        self.expression_matrix_1 = expression_matrix_1
        self.cell_labels_1 = cell_labels_1
        self.expression_matrix_2 = expression_matrix_2
        self.cell_labels_2 = cell_labels_2
        self.top_genes = top_genes
        self.cell_name_column = cell_name_column
        self.cell_type_column = cell_type_column
        self.mode_mapping = {"target": 1, "query": 2, "train": 3}
        if mode not in self.mode_mapping:
            raise ValueError(f"Unsupported mode: {mode}. Choose from {list(self.mode_mapping.keys())}")
        self.mode_str = mode
        self.mode = self.mode_mapping[mode]

    def process_data(self):
        """
        处理基因表达矩阵和细胞标签，提取高表达基因，并更新细胞标签。

        Returns:
            y (list): 每个细胞的高表达基因列表。
            z (list): 唯一的细胞 ID 列表，格式为 [[mode, number], ...]。
            updated_cell_labels_1 (pd.DataFrame): 更新后的第一个数据集细胞标签。
            updated_cell_labels_2 (pd.DataFrame, optional): 更新后的第二个数据集细胞标签，如果存在。
        """
        if self.expression_matrix_2 is not None and self.cell_labels_2 is not None:
            return self._process_two_datasets()
        else:
            return self._process_single_dataset()

    def _process_single_dataset(self):
        """
        处理单个数据集。
        """
        common_cells = self.expression_matrix_1.columns.intersection(self.cell_labels_1[self.cell_name_column])
        if len(common_cells) == 0:
            raise ValueError(f"No matching cells between expression_matrix_1 and cell_labels_1 using column '{self.cell_name_column}'.")
        
        # 提取非 MT^ 开头的基因
        filtered_matrix = self.expression_matrix_1.loc[~self.expression_matrix_1.index.str.startswith("MT^")]
        
        # 获取每个细胞的高表达基因
        y = []
        for cell in common_cells:
            top_genes = filtered_matrix[cell].sort_values(ascending=False).head(self.top_genes)
            y.append(top_genes.index.tolist())
        
        # 为每个细胞生成唯一 ID (数字_数字的形式)，并将其转换为 [mode, number] 格式
        z = [[self.mode, i+1] for i in range(len(common_cells))]
        
        # 更新 cell_labels 数据
        updated_cell_labels = self.cell_labels_1[self.cell_labels_1[self.cell_name_column].isin(common_cells)].copy()
        updated_cell_labels["cell_id"] = [f"{self.mode}_{i+1}" for i in range(len(common_cells))]
        updated_cell_labels["mode"] = self.mode_str
        
        # 如果是 train 模式，添加 target 和 cell_type 编码
        if self.mode_str == "train":
            if self.cell_type_column not in updated_cell_labels.columns:
                updated_cell_labels[self.cell_type_column] = ''
            updated_cell_labels["cell_type_code"] = updated_cell_labels[self.cell_type_column].astype("category").cat.codes
            updated_cell_labels["cell_type_code"] = '3_' + updated_cell_labels["cell_type_code"].astype(str)
            z = updated_cell_labels["cell_type_code"].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]).tolist()

        return y, z, updated_cell_labels

    def _process_two_datasets(self):
        """
        处理两个数据集，并返回更新后的标签和编码。
        """
        # 获取两个数据集的 cell_type 列并计算并集
        all_cell_types = pd.concat([self.cell_labels_1[self.cell_type_column], self.cell_labels_2[self.cell_type_column]]).unique()
        type_mapping = {cell_type: f"3_{i}" for i, cell_type in enumerate(sorted(all_cell_types))}

        updated_cell_labels_1 = self.cell_labels_1.copy()
        updated_cell_labels_2 = self.cell_labels_2.copy()

        updated_cell_labels_1["cell_type_code"] = updated_cell_labels_1[self.cell_type_column].map(type_mapping)
        updated_cell_labels_2["cell_type_code"] = updated_cell_labels_2[self.cell_type_column].map(type_mapping)

        updated_cell_labels_1["cell_id"] = [f"3_{i+1}" for i in range(len(updated_cell_labels_1))]
        updated_cell_labels_2["cell_id"] = [f"3_{i+1}" for i in range(len(updated_cell_labels_2))]
        
        y_1, y_2 = [], []
        common_cells_1 = self.expression_matrix_1.columns.intersection(updated_cell_labels_1[self.cell_name_column])
        common_cells_2 = self.expression_matrix_2.columns.intersection(updated_cell_labels_2[self.cell_name_column])

        filtered_matrix1 = self.expression_matrix_1.loc[~self.expression_matrix_1.index.str.startswith("MT^")]
        filtered_matrix2 = self.expression_matrix_2.loc[~self.expression_matrix_2.index.str.startswith("MT^")]
        
        for cell in common_cells_1:
            top_genes = filtered_matrix1[cell].sort_values(ascending=False).head(self.top_genes)
            y_1.append(top_genes.index.tolist())

        for cell in common_cells_2:
            top_genes = filtered_matrix2[cell].sort_values(ascending=False).head(self.top_genes)
            y_2.append(top_genes.index.tolist())

        # z_1 = updated_cell_labels_1["cell_type_code"].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]).tolist()
        # z_2 = updated_cell_labels_2["cell_type_code"].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]).tolist()

        z_1 = [list(map(int, x.split('_'))) for x in updated_cell_labels_1["cell_type_code"]]
        z_2 = [list(map(int, x.split('_'))) for x in updated_cell_labels_2["cell_type_code"]]

        return y_1, y_2, z_1, z_2, updated_cell_labels_1, updated_cell_labels_2

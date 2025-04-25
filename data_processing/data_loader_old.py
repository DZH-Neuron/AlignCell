import pandas as pd

class DataLoader:
    def __init__(self, expression_matrix, cell_labels, top_genes=1000, mode="target", cell_name_column='cell_name', cell_type_column='cell_type'):
        """
        初始化数据加载器
        
        Args:
            expression_matrix (pd.DataFrame): 基因表达矩阵，行是基因名，列是细胞名。
            cell_labels (pd.DataFrame): 细胞标签，包含列用于指定细胞名称和标签。
            top_genes (int): 每个细胞提取的高表达基因数目。
            mode (str): 当前数据模式，支持 'target', 'query', 或 'train'。
            cell_name_column (str): 用于匹配细胞名称的列名，默认是 'cell_name'。
            cell_type_column (str): 用于匹配细胞类型的列名，默认是 'cell_type'。
        """
        # 检查输入数据格式
        if not isinstance(expression_matrix, pd.DataFrame):
            raise TypeError("expression_matrix should be a pandas DataFrame.")
        if not isinstance(cell_labels, pd.DataFrame):
            raise TypeError("cell_labels should be a pandas DataFrame.")
        
        self.expression_matrix = expression_matrix
        self.cell_labels = cell_labels
        self.top_genes = top_genes
        self.cell_name_column = cell_name_column
        self.cell_type_column = cell_type_column  # 新增的细胞类型列参数
        
        # 模式映射：字符串 -> 数字（内部使用，但不影响最终输出）
        self.mode_mapping = {"target": 1, "query": 2, "train": 3}
        if mode not in self.mode_mapping:
            raise ValueError(f"Unsupported mode: {mode}. Choose from {list(self.mode_mapping.keys())}")
        
        self.mode_str = mode  # 保留原始字符串模式
        self.mode = self.mode_mapping[mode]  # 内部存储为数字模式

    def process_data(self):
        """
        处理基因表达矩阵和细胞标签，提取高表达基因，并更新细胞标签。

        Returns:
            y (list): 每个细胞的高表达基因列表。
            z (list): 唯一的细胞 ID 列表，格式为 [[mode, number], ...]。
            updated_cell_labels (pd.DataFrame): 更新后的细胞标签，包含 'cell_name'、'label'、'mode'、'cell_id' 和 'cell_type_code'。
        """
        # 检查表达矩阵和细胞标签的匹配
        common_cells = self.expression_matrix.columns.intersection(self.cell_labels[self.cell_name_column])
        if len(common_cells) == 0:
            raise ValueError(f"No matching cells between expression_matrix and cell_labels using column '{self.cell_name_column}'.")
        
        # 提取非 MT^ 开头的基因
        filtered_matrix = self.expression_matrix.loc[~self.expression_matrix.index.str.startswith("MT^")]
        
        # 获取每个细胞的高表达基因
        y = []
        for cell in common_cells:
            # 按表达量排序，选择前 top_genes 个基因
            top_genes = filtered_matrix[cell].sort_values(ascending=False).head(self.top_genes)
            y.append(top_genes.index.tolist())
        
        # 为每个细胞生成唯一 ID (数字_数字的形式)，并将其转换为 [mode, number] 格式
        z = [[self.mode, i+1] for i in range(len(common_cells))]
        
        # 更新 cell_labels 数据
        updated_cell_labels = self.cell_labels[self.cell_labels[self.cell_name_column].isin(common_cells)].copy()
        updated_cell_labels["cell_id"] = [f"{self.mode}_{i+1}" for i in range(len(common_cells))]
        updated_cell_labels["mode"] = self.mode_str  # 添加原始字符串模式
        
        # 如果是 train 模式，添加 target 和 cell_type 编码
        if self.mode_str == "train":
            # 编码 target 列
            # 如果缺少 cell_type 列，初始化为空值
            if self.cell_type_column not in updated_cell_labels.columns:
                updated_cell_labels[self.cell_type_column] = ''
            
            # 编码 cell_type 列
            updated_cell_labels["cell_type_code"] = updated_cell_labels[self.cell_type_column].astype("category").cat.codes
            
            updated_cell_labels["cell_type_code"] = '3_' + updated_cell_labels["cell_type_code"].astype(str)
            z = updated_cell_labels["cell_type_code"].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]).tolist()
            

        return y, z, updated_cell_labels


class DataLoader2:
    def __init__(self, expression_matrix_1, cell_labels_1, expression_matrix_2, cell_labels_2, top_genes=1000, cell_name_column='cell_name', cell_type_column='cell_type'):
        """
        初始化数据加载器，支持两个数据集的并集编码。
        
        Args:
            expression_matrix_1 (pd.DataFrame): 第一个基因表达矩阵，行是基因名，列是细胞名。
            cell_labels_1 (pd.DataFrame): 第一个细胞标签，包含列用于指定细胞名称和类型。
            expression_matrix_2 (pd.DataFrame): 第二个基因表达矩阵，行是基因名，列是细胞名。
            cell_labels_2 (pd.DataFrame): 第二个细胞标签，包含列用于指定细胞名称和类型。
            top_genes (int): 每个细胞提取的高表达基因数目。
            cell_name_column (str): 用于匹配细胞名称的列名，默认是 'cell_name'。
            cell_type_column (str): 用于匹配细胞类型的列名，默认是 'cell_type'。
        """
        # 检查输入数据格式
        if not isinstance(expression_matrix_1, pd.DataFrame) or not isinstance(expression_matrix_2, pd.DataFrame):
            raise TypeError("expression_matrix should be a pandas DataFrame.")
        if not isinstance(cell_labels_1, pd.DataFrame) or not isinstance(cell_labels_2, pd.DataFrame):
            raise TypeError("cell_labels should be a pandas DataFrame.")
        
        self.expression_matrix_1 = expression_matrix_1
        self.cell_labels_1 = cell_labels_1
        self.expression_matrix_2 = expression_matrix_2
        self.cell_labels_2 = cell_labels_2
        self.top_genes = top_genes
        self.cell_name_column = cell_name_column
        self.cell_type_column = cell_type_column

    def process_data(self):
        """
        处理两个基因表达矩阵和细胞标签，提取高表达基因，并更新细胞标签的并集编码。

        Returns:
            y (list): 每个细胞的高表达基因列表。
            z (list): 唯一的细胞 ID 列表，格式为 [[3, number], ...]。
            updated_cell_labels_1 (pd.DataFrame): 更新后的第一个数据集细胞标签。
            updated_cell_labels_2 (pd.DataFrame): 更新后的第二个数据集细胞标签。
        """
        # 获取两个数据集的 cell_type 列并计算并集
        all_cell_types = pd.concat([self.cell_labels_1[self.cell_type_column], self.cell_labels_2[self.cell_type_column]]).unique()
        # 为并集中的每个 cell_type 生成唯一编码
        type_mapping = {cell_type: f"3_{i}" for i, cell_type in enumerate(sorted(all_cell_types))}

        # 更新 cell_labels 数据
        updated_cell_labels_1 = self.cell_labels_1.copy()
        updated_cell_labels_2 = self.cell_labels_2.copy()

        updated_cell_labels_1["cell_type_code"] = updated_cell_labels_1[self.cell_type_column].map(type_mapping)
        updated_cell_labels_2["cell_type_code"] = updated_cell_labels_2[self.cell_type_column].map(type_mapping)

        # 为每个细胞生成唯一 ID (3_数字的形式)
        updated_cell_labels_1["cell_id"] = [f"3_{i+1}" for i in range(len(updated_cell_labels_1))]
        updated_cell_labels_2["cell_id"] = [f"3_{i+1}" for i in range(len(updated_cell_labels_2))]

        # 提取每个细胞的高表达基因
        y_1, y_2 = [], []
        common_cells_1 = self.expression_matrix_1.columns.intersection(updated_cell_labels_1[self.cell_name_column])
        common_cells_2 = self.expression_matrix_2.columns.intersection(updated_cell_labels_2[self.cell_name_column])

        # 获取每个细胞的高表达基因
        for cell in common_cells_1:
            top_genes = self.expression_matrix_1[cell].sort_values(ascending=False).head(self.top_genes)
            y_1.append(top_genes.index.tolist())

        for cell in common_cells_2:
            top_genes = self.expression_matrix_2[cell].sort_values(ascending=False).head(self.top_genes)
            y_2.append(top_genes.index.tolist())

        # 返回结果
        return y_1, y_2, updated_cell_labels_1, updated_cell_labels_2

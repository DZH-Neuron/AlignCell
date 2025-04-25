import pandas as pd

def split_cell(updated_cell_labels, expression_matrix, group_column, cell_name_column='cell_name', train_ratio=0.8, random_state=42):
    """
    按照指定列（如 'cell_types'）进行分组，并将每个组的数据按指定比例划分为训练集和测试集，同时更新基因表达矩阵的训练集和测试集。

    Args:
        updated_cell_labels (pd.DataFrame): 包含细胞名称、类型和其它相关信息的数据框。
        expression_matrix (pd.DataFrame): 基因表达矩阵，行是基因名，列是细胞名。
        group_column (str): 用于分组的列名，例如 'cell_types'。
        cell_name_column (str): 用于指定细胞名称的列名，默认是 'cell_name'。
        train_ratio (float): 训练集占总数据的比例，默认是 0.8。
        random_state (int): 随机种子，保证每次划分一致，默认是 42。

    Returns:
        train_data (pd.DataFrame): 划分后的训练集。
        test_data (pd.DataFrame): 划分后的测试集。
        train_expression_matrix (pd.DataFrame): 划分后的训练集基因表达矩阵。
        test_expression_matrix (pd.DataFrame): 划分后的测试集基因表达矩阵。
    """
    # 用于存储训练集和测试集的列表
    df1_list = []  # 训练集
    df2_list = []  # 测试集

    # 按指定列分组
    groups = updated_cell_labels.groupby(group_column)

    # 遍历每个组
    for name, group in groups:
        # 打乱组内数据的顺序
        shuffled_group = group.sample(frac=1, random_state=random_state)
        
        # 划分每个组的 80% 和 20%
        split_index = int(len(shuffled_group) * train_ratio)
        group_train = shuffled_group.iloc[:split_index]
        group_test = shuffled_group.iloc[split_index:]
        
        # 将每个组的训练集和测试集加入到对应的列表
        df1_list.append(group_train)
        df2_list.append(group_test)

    # 合并训练集和测试集
    train_data = pd.concat(df1_list, axis=0).reset_index(drop=True)
    test_data = pd.concat(df2_list, axis=0).reset_index(drop=True)

    # 根据划分后的 train_data 和 test_data 获取相应的细胞名称
    train_cells = train_data[cell_name_column].tolist()
    test_cells = test_data[cell_name_column].tolist()

    # 使用基因表达矩阵中的细胞名称来提取对应列
    train_expression_matrix = expression_matrix[train_cells]
    test_expression_matrix = expression_matrix[test_cells]

    return train_data, test_data, train_expression_matrix, test_expression_matrix
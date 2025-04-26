import pandas as pd

def split_cell(updated_cell_labels, expression_matrix, group_column, cell_name_column='cell_name', train_ratio=0.8, random_state=42):
 """
    Group the data by a specified column (e.g., 'cell_types') and divide each group into a training set and a test set based on a given ratio, while updating the gene expression matrices for the training and test sets.

    Args:
        updated_cell_labels (pd.DataFrame): A DataFrame containing cell names, types, and other relevant information.
        expression_matrix (pd.DataFrame): Gene expression matrix, with genes as rows and cells as columns.
        group_column (str): The column name used for grouping, e.g., 'cell_types'.
        cell_name_column (str): The column name used to specify cell names. Default is 'cell_name'.
        train_ratio (float): The proportion of data allocated to the training set. Default is 0.8.
        random_state (int): Random seed to ensure reproducibility of the splits. Default is 42.

    Returns:
        train_data (pd.DataFrame): The resulting training set.
        test_data (pd.DataFrame): The resulting test set.
        train_expression_matrix (pd.DataFrame): The gene expression matrix for the training set.
        test_expression_matrix (pd.DataFrame): The gene expression matrix for the test set.
    """
    # A list to store the training set and test set.
    df1_list = []  # Training set
    df2_list = []  # Test set

    # Group by the specified column
    groups = updated_cell_labels.groupby(group_column)

    # Iterate through each group
    for name, group in groups:
        # Shuffle the data within each group
        shuffled_group = group.sample(frac=1, random_state=random_state)
        
        split_index = int(len(shuffled_group) * train_ratio)
        group_train = shuffled_group.iloc[:split_index]
        group_test = shuffled_group.iloc[split_index:]
        
        # Add the training and test sets of each group to the corresponding lists.
        df1_list.append(group_train)
        df2_list.append(group_test)

    # Merge the training set and test set.
    train_data = pd.concat(df1_list, axis=0).reset_index(drop=True)
    test_data = pd.concat(df2_list, axis=0).reset_index(drop=True)

    # Get the corresponding cell names from the divided train_data and test_data.
    train_cells = train_data[cell_name_column].tolist()
    test_cells = test_data[cell_name_column].tolist()

    # Use the cell names from the gene expression matrix to extract the corresponding columns.
    train_expression_matrix = expression_matrix[train_cells]
    test_expression_matrix = expression_matrix[test_cells]

    return train_data, test_data, train_expression_matrix, test_expression_matrix

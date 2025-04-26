import pandas as pd

class DataLoader:
    def __init__(self, expression_matrix_1, cell_labels_1, expression_matrix_2=None, cell_labels_2=None, top_genes=1000, mode="target", cell_name_column='cell_name', cell_type_column='cell_type'):
        """
        Initialize the data loader, supporting the processing of one or two datasets.
            Args:
                expression_matrix_1 (pd.DataFrame): The first gene expression matrix, with genes as rows and cells as columns.
                cell_labels_1 (pd.DataFrame): Cell labels for the first dataset, including columns specifying cell names and cell types.
                expression_matrix_2 (pd.DataFrame, optional): The second gene expression matrix, with genes as rows and cells as columns. Default is None.
                cell_labels_2 (pd.DataFrame, optional): Cell labels for the second dataset, including columns specifying cell names and cell types. Default is None.
                top_genes (int): The number of highly expressed genes to extract for each cell.
                mode (str): The current data mode, supporting 'target', 'query', or 'train'.
                cell_name_column (str): The column name used to match cell names. Default is 'cell_name'.
                cell_type_column (str): The column name used to match cell types. Default is 'cell_type'.
        """
        # Check the input data format.
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
        Process gene expression matrices and cell labels, extract highly expressed genes, and update cell labels.

            Returns:
                y (list): A list of highly expressed genes for each cell.
                z (list): A list of unique cell IDs, formatted as [[mode, number], ...].
                updated_cell_labels_1 (pd.DataFrame): Updated cell labels for the first dataset.
                updated_cell_labels_2 (pd.DataFrame, optional): Updated cell labels for the second dataset, if provided.
        """
        if self.expression_matrix_2 is not None and self.cell_labels_2 is not None:
            return self._process_two_datasets()
        else:
            return self._process_single_dataset()

    def _process_single_dataset(self):
        """
        Process a single dataset.
        """
        common_cells = self.expression_matrix_1.columns.intersection(self.cell_labels_1[self.cell_name_column])
        if len(common_cells) == 0:
            raise ValueError(f"No matching cells between expression_matrix_1 and cell_labels_1 using column '{self.cell_name_column}'.")
        
        filtered_matrix = self.expression_matrix_1.loc[~self.expression_matrix_1.index.str.startswith("MT^")]
        
        y = []
        for cell in common_cells:
            top_genes = filtered_matrix[cell].sort_values(ascending=False).head(self.top_genes)
            y.append(top_genes.index.tolist())
        
        # Generate a unique ID for each cell in the format of number_number, and convert it into [mode, number] format.
        z = [[self.mode, i+1] for i in range(len(common_cells))]
        
        # Update the cell_labels data.
        updated_cell_labels = self.cell_labels_1[self.cell_labels_1[self.cell_name_column].isin(common_cells)].copy()
        updated_cell_labels["cell_id"] = [f"{self.mode}_{i+1}" for i in range(len(common_cells))]
        updated_cell_labels["mode"] = self.mode_str
        
        # If in 'train' mode, add target and cell type encoding.
        if self.mode_str == "train":
            if self.cell_type_column not in updated_cell_labels.columns:
                updated_cell_labels[self.cell_type_column] = ''
            updated_cell_labels["cell_type_code"] = updated_cell_labels[self.cell_type_column].astype("category").cat.codes
            updated_cell_labels["cell_type_code"] = '3_' + updated_cell_labels["cell_type_code"].astype(str)
            z = updated_cell_labels["cell_type_code"].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]).tolist()

        return y, z, updated_cell_labels

    def _process_two_datasets(self):
        """
        Process two datasets and return the updated labels and encodings.
        """
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

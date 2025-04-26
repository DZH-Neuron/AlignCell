import pandas as pd

class DataLoader:
    def __init__(self, expression_matrix, cell_labels, top_genes=1000, mode="target", cell_name_column='cell_name', cell_type_column='cell_type'):
        if not isinstance(expression_matrix, pd.DataFrame):
            raise TypeError("expression_matrix should be a pandas DataFrame.")
        if not isinstance(cell_labels, pd.DataFrame):
            raise TypeError("cell_labels should be a pandas DataFrame.")
        
        self.expression_matrix = expression_matrix
        self.cell_labels = cell_labels
        self.top_genes = top_genes
        self.cell_name_column = cell_name_column
        self.cell_type_column = cell_type_column 
        
        self.mode_mapping = {"target": 1, "query": 2, "train": 3}
        if mode not in self.mode_mapping:
            raise ValueError(f"Unsupported mode: {mode}. Choose from {list(self.mode_mapping.keys())}")
        
        self.mode_str = mode
        self.mode = self.mode_mapping[mode] 

    def process_data(self):
        common_cells = self.expression_matrix.columns.intersection(self.cell_labels[self.cell_name_column])
        if len(common_cells) == 0:
            raise ValueError(f"No matching cells between expression_matrix and cell_labels using column '{self.cell_name_column}'.")
        
        filtered_matrix = self.expression_matrix.loc[~self.expression_matrix.index.str.startswith("MT^")]
        
        y = []
        for cell in common_cells:
            top_genes = filtered_matrix[cell].sort_values(ascending=False).head(self.top_genes)
            y.append(top_genes.index.tolist())
        
        z = [[self.mode, i+1] for i in range(len(common_cells))]
        
        updated_cell_labels = self.cell_labels[self.cell_labels[self.cell_name_column].isin(common_cells)].copy()
        updated_cell_labels["cell_id"] = [f"{self.mode}_{i+1}" for i in range(len(common_cells))]
        updated_cell_labels["mode"] = self.mode_str
        
        if self.mode_str == "train":
            if self.cell_type_column not in updated_cell_labels.columns:
                updated_cell_labels[self.cell_type_column] = ''
            
            updated_cell_labels["cell_type_code"] = updated_cell_labels[self.cell_type_column].astype("category").cat.codes
            
            updated_cell_labels["cell_type_code"] = '3_' + updated_cell_labels["cell_type_code"].astype(str)
            z = updated_cell_labels["cell_type_code"].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]).tolist()
            

        return y, z, updated_cell_labels


class DataLoader2:
    def __init__(self, expression_matrix_1, cell_labels_1, expression_matrix_2, cell_labels_2, top_genes=1000, cell_name_column='cell_name', cell_type_column='cell_type'):
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

        for cell in common_cells_1:
            top_genes = self.expression_matrix_1[cell].sort_values(ascending=False).head(self.top_genes)
            y_1.append(top_genes.index.tolist())

        for cell in common_cells_2:
            top_genes = self.expression_matrix_2[cell].sort_values(ascending=False).head(self.top_genes)
            y_2.append(top_genes.index.tolist())

        return y_1, y_2, updated_cell_labels_1, updated_cell_labels_2

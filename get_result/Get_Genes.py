import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def extract_high_attention_genes(
    slc_attention,
    gene_name,
    preprocess,
    ref_cell_labels,
    threshold=0.0015,
    cell_index_column='Unnamed: 0',
    cell_type_column='Type',
    group_column=None  # For example, 'species' or 'modality', optional.
):
    """
    Extract highly attended genes and calculate the occurrence count and average attention value for each group-celltype.

    Args:
        slc_attention: np.ndarray (num_cells, seq_len)
        gene_name: np.ndarray (num_cells, seq_len)
        preprocess: An object with idx2word mapping
        ref_cell_labels: pd.DataFrame containing cell types, indices, and group information
        threshold: float, attention filtering threshold
        cell_index_column: str, the column name for cell index
        cell_type_column: str, the column name for cell type
        group_column: str or None, the column name for grouping (e.g., species or modality). If None, no grouping is performed.

    Returns:
        pd.DataFrame, with columns ['group', 'celltype', 'gene_symbol', 'count', 'mean_attention'], or without 'group'.
    """

    all_gene_indices = []
    all_gene_attentions = []

    for query_index in range(slc_attention.shape[0]):
        attention_values = slc_attention[query_index]
        indices = np.where(attention_values > threshold)[0]
        indices_plus_1 = indices + 1

        gene_indices_for_query = gene_name[query_index, indices_plus_1]
        gene_attentions_for_query = attention_values[indices]

        all_gene_indices.append(gene_indices_for_query)
        all_gene_attentions.append(gene_attentions_for_query)

    all_gene_names = []
    all_gene_name_attn_pairs = []

    for gene_indices_for_query, attn_values in zip(all_gene_indices, all_gene_attentions):
        gene_names = [preprocess.idx2word[idx] for idx in gene_indices_for_query]
        all_gene_names.append(gene_names)
        all_gene_name_attn_pairs.append(list(zip(gene_names, attn_values)))

    index_to_gene_names = {
        ref_cell_labels.loc[i, cell_index_column]: all_gene_names[i]
        for i in range(len(ref_cell_labels))
    }
    index_to_gene_name_attn = {
        ref_cell_labels.loc[i, cell_index_column]: all_gene_name_attn_pairs[i]
        for i in range(len(ref_cell_labels))
    }

    celltype_to_gene_names_count = {}
    celltype_to_gene_attn_sum = defaultdict(lambda: defaultdict(float))
    celltype_to_gene_attn_count = defaultdict(lambda: defaultdict(int))

    group_values = ref_cell_labels[group_column].unique() if group_column else [None]
    expanded_data = []

    for group in group_values:
        if group is not None:
            group_df = ref_cell_labels[ref_cell_labels[group_column] == group]
        else:
            group_df = ref_cell_labels

        for celltype in group_df[cell_type_column].unique():
            celltype_gene_names_count = Counter()

            for idx, row in group_df[group_df[cell_type_column] == celltype].iterrows():
                index = row[cell_index_column]
                gene_names = index_to_gene_names.get(index, [])
                gene_attn_pairs = index_to_gene_name_attn.get(index, [])

                celltype_gene_names_count.update(gene_names)

                for gene_name, attn_value in gene_attn_pairs:
                    celltype_to_gene_attn_sum[(group, celltype)][gene_name] += attn_value
                    celltype_to_gene_attn_count[(group, celltype)][gene_name] += 1

            sorted_gene_names = sorted(celltype_gene_names_count.items(), key=lambda x: x[1], reverse=True)
            celltype_to_gene_names_count[(group, celltype)] = sorted_gene_names

            for gene_symbol, count in sorted_gene_names:
                total_attn = celltype_to_gene_attn_sum[(group, celltype)][gene_symbol]
                attn_count = celltype_to_gene_attn_count[(group, celltype)][gene_symbol]
                mean_attention = total_attn / attn_count if attn_count > 0 else 0

                row_data = (group, celltype, gene_symbol, count, mean_attention) if group is not None else (celltype, gene_symbol, count, mean_attention)
                expanded_data.append(row_data)

    columns = ['group', 'celltype', 'gene_symbol', 'count', 'mean_attention'] if group_column else ['celltype', 'gene_symbol', 'count', 'mean_attention']
    expanded_gene_counts_df = pd.DataFrame(expanded_data, columns=columns)

    return expanded_gene_counts_df



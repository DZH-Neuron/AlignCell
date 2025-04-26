import os
import torch
import argparse
import numpy as np
from AlignCell.model import TransformerEncoder
from AlignCell.train import train
from data_processing.data_loader import load_training_data
from torch.utils.data import random_split
from data_processing.preprocess import Preprocess_gene
from torch import nn

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train AlignCell Model")
    parser.add_argument('--train_data', type=str, required=True, help="Path to training data")
    parser.add_argument('--w2v_model', type=str, default='./data_processing/hum_dic_gene2vec.model',required=True, help="Path to pre-trained Word2Vec model")
    parser.add_argument('--model_dir', type=str, default='./', help="Model checkpoint directory")
    parser.add_argument('--sen_len', type=int, default=1000, help="Maximum sentence length")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epoch', type=int, default=350, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=0.000005, help="Learning rate")
    return parser.parse_args()

def load_and_preprocess_data(train_data_path, w2v_model_path, sen_len):
    """Load and preprocess data"""
    print("Loading data...")
    gene_name, gene_label = load_training_data(train_data_path)
    
    preprocess = Preprocess_gene(gene_name, sen_len, w2v_path=w2v_model_path)
    embedding = preprocess.make_embedding()
    gene_name = preprocess.sentence_word2idx()
    gene_label = preprocess.labels_to_tensor(gene_label)
    
    return gene_name, gene_label, embedding

def create_model(input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding, fast_attention_config):
    """Create and return the model"""
    model = TransformerEncoder(input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding, fast_attention_config)
    return model

def split_data(gene_name, gene_label, train_size=0.8):
    """Split the dataset"""
    train_size = int(train_size * len(gene_name))
    val_size = len(gene_name) - train_size

    all_dataset = TwitterDataset(x=gene_name, z=gene_label)
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20)

    return train_loader, val_loader

def main():
    # Parse command-line arguments
    args = parse_args()

    # Check the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading and preprocessing
    gene_name, gene_label, embedding = load_and_preprocess_data(args.train_data, args.w2v_model, args.sen_len)

    # Model parameter settings
    input_dim ,d_model = embedding.size()
    num_layers = 6
    num_heads = 8
    d_ff = 1000
    max_len = 1001
    dropout = 0
    fast_attention_config = {
        'nb_features': None,
        'ortho_scaling': 0,
        'causal': False,
        'generalized_attention': False,
        'kernel_fn': nn.ReLU(),
        'no_projection': False
    }
    fix_embedding = False

    # Create the model
    model = create_model(input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding, fast_attention_config)
    model = model.to(device)

    # Dataset splitting
    train_loader, val_loader = split_data(gene_name, gene_label)

    # Start training
    print("Starting training...")
    training(args.batch_size, args.epoch, args.lr, args.model_dir, train_loader, val_loader, model, device)

if __name__ == "__main__":
    main()

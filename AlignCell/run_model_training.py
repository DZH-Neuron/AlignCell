import torch
import torch.nn as nn
import os
from torch.utils.data import random_split
from AlignCell import TransformerEncoder, PositionalEncoding, EncoderLayer, FeedForward

def create_model(embedding, input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding, fast_attention_config):
    """Create and return the model"""
    model = TransformerEncoder(embedding, input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding, fast_attention_config)
    return model

def split_data(all_dataset, batch_size, size_ratio=0.8):
    """Split the dataset"""
    train_size = int(size_ratio * len(all_dataset))
    val_size = len(all_dataset) - train_size

    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)

    return train_loader, val_loader

def run_model_training(
    embedding,
    training,
    all_dataset,
    batch_size=64,
    epochs=200,
    learning_rate=0.000005,
    model_dir="./model",
    pretrained_model_path=None,
    device=None,
    num_layers=6,
    num_heads=8,
    d_ff=1000,
    max_len=1001,
    dropout=0,
    fix_embedding=False,
    fast_attention_config=None,
    output_attentions=False,
):
    """
    A reusable function to initialize, load or fine-tune a model.

    Parameters:
        embedding (torch.Tensor): The embedding tensor to define input dimensions.
        training (callable): Function to train the model.
        all_dataset (list or set).
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of epochs for training. Default is 10.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        model_dir (str): Directory to save the trained model. Default is './model'.
        pretrained_model_path (str, optional): Path to a pretrained model. If provided, load it. Default is None.
        device (torch.device, optional): Device to run the model on. Default is auto-detected.
        num_layers (int): Number of layers in the model. Default is 6.
        num_heads (int): Number of attention heads. Default is 8.
        d_ff (int): Dimension of feedforward layer. Default is 1000.
        max_len (int): Maximum sequence length. Default is 1001.
        dropout (float): Dropout rate. Default is 0.
        fix_embedding (bool): Whether to fix embedding weights. Default is False.
        fast_attention_config (dict, optional): Configuration for fast attention mechanism. Default is None.
        output_attentions: Output attention weight. Default is False.

    Returns:
        None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fast_attention_config is None:
        fast_attention_config = {
            'nb_features': None,
            'ortho_scaling': 0,
            'causal': False,
            'generalized_attention': False,
            'kernel_fn': nn.ReLU(),
            'no_projection': False
        }

    # Set input dimensions
    input_dim, d_model = embedding.size()

    # Initialize or load the model
    try:
        # Initialize or load the model
        if pretrained_model_path:
            print(f"Loading pretrained model from {pretrained_model_path}...")
            model = torch.load(pretrained_model_path, weights_only=False)
        else:
            print("Initializing new model...")
            model = create_model(
                embedding=embedding,
                input_dim=input_dim,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                max_len=max_len,
                dropout=dropout,
                fix_embedding=fix_embedding,
                fast_attention_config=fast_attention_config
            )

        model = model.to(device)

        # Split data into training and validation
        train_loader, val_loader = split_data(all_dataset, batch_size=batch_size)

        os.makedirs(model_dir, exist_ok=True)

        # Train the model
        print("Starting training...")
        training(
            batch_size=batch_size,
            n_epoch=epochs,
            lr=learning_rate,
            model_dir=model_dir,
            train=train_loader,
            valid=val_loader,
            model=model,
            device=device,
            output_attentions=output_attentions
        )

    finally:
        # Release memory
        # del model
        torch.cuda.empty_cache()
        print("Model training finished, GPU memory cleared.")


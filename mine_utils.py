
import matplotlib.pyplot as plt
import wandb
import umap
import torch
from torch import nn
import seaborn as sns
import os
from sklearn.manifold import TSNE
import random 
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from models.heads.MLP import MLP
from models.heads.FCN import FCN



def select_head(head_model, head_config):
    if head_model == "MLP":
        in_dim, hidden_dim, output_size = head_config["layers_config"]
        model = MLP(in_dim, hidden_dim, output_size)
    elif head_model == "FCN":
        in_dim, out_dim = head_config["layers_config"]
        model = FCN(in_dim, out_dim)

    model.to(head_config['gpu'])
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=head_config['lr'])
    criterion = nn.CrossEntropyLoss()
    return model, adam_optimizer, criterion


def plot_umap(embedding, train_labels, name, save_path="umap_plot.pdf"):
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], 
                         embedding[:, 1], 
                         c=train_labels.numpy(), 
                         cmap='Spectral', 
                         edgecolor='k',  
                         alpha=0.7,  
                         s=50)  

    ax.set_title("UMAP Projection", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Labels", fontsize=10, color="gray", alpha=0.7)

    plt.savefig(save_path, format='pdf', bbox_inches='tight')  # Save as PDF
    plt.close(fig)

    wandb.log({f'{name}': fig})  # Log the saved PDF
    return fig
    

def umap_embedding(model, dataloader, device, head_model=None, mtype='ssl', name='ssl_before_finetuning', save_path="."):
    train_emb = []
    train_labels = []

    model.eval()
    batch_samples = next(iter(dataloader))[0].shape[0]

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if x.shape[0] != batch_samples:
                continue
            z_emb = model.linear_prob(x)
            if mtype != 'ssl':
                z_emb = head_model(z_emb)
            z_emb = z_emb.cpu().numpy()
            labels = y.cpu().numpy()
            train_emb.append(z_emb)
            train_labels.append(labels)

    train_emb = torch.cat([torch.tensor(e) for e in train_emb], dim=0)
    train_labels = torch.cat([torch.tensor(l) for l in train_labels], dim=0)
    torch.save(train_emb, os.path.join(save_path, name + '.pth'))
    mapper = umap.UMAP(n_neighbors=5, 
                       min_dist=0.2,    
                       metric='euclidean', 
                       random_state=42,
                       transform_seed=42).fit(train_emb.numpy())

    return mapper.embedding_, train_labels


def tsne_embedding(model, dataloader, device, head_model=None, mtype='ssl', name='ssl_before_finetuning', save_path="."):
    train_emb = []
    train_labels = []

    model.eval()
    batch_samples = next(iter(dataloader))[0].shape[0]

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if x.shape[0] != batch_samples:
                continue
            z_emb = model.linear_prob(x)
            if mtype == 'mlp':
                z_emb = head_model(z_emb)
            z_emb = z_emb.cpu().numpy()
            labels = y.cpu().numpy()
            train_emb.append(z_emb)
            train_labels.append(labels)

    train_emb = torch.cat([torch.tensor(e) for e in train_emb], dim=0)
    train_labels = torch.cat([torch.tensor(l) for l in train_labels], dim=0)
    torch.save(train_emb, os.path.join(save_path, name + '.pth'))
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='pca', random_state=10, perplexity=3).fit_transform(train_emb.numpy())
    return X_embedded, train_labels


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the head model to use', required=True)
    parser.add_argument('--description', type=str, help='Description of the experiment', required=True)
    parser.add_argument('--head_configuration_file', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--encoder_configuration_file', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for the experiment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the head model')
    parser.add_argument('--encoder_checkpoint_path', type=str, help='Path to the encoder checkpoint file', required=True)
    parser.add_argument('--is_finetuning', type=int, help='set to 1 to use linear probing, 0 to train the encoder as well', required=True)
    parser.add_argument('--seed', type=int, default=10, help='Seed for random number')
    return parser

def load_fragment_dataset(batch_size):
    train_data = torch.load('Dataset/Fragment/ecg-fragment_360hz/train.pt')
    test_data = torch.load('Dataset/Fragment/ecg-fragment_360hz/test.pt')
    val_data = torch.load('Dataset/Fragment/ecg-fragment_360hz/val.pt')

    x_train_data, y_train_data = train_data['samples'], train_data['labels']
    x_test_data, y_test_data = test_data['samples'], test_data['labels']
    x_val_data, y_val_data = val_data['samples'], val_data['labels']

    train_loader = DataLoader(TensorDataset(x_train_data, y_train_data), 
                                            batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_data, y_test_data), 
                                            batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val_data, y_val_data), 
                                            batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader
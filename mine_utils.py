
import matplotlib.pyplot as plt
import wandb
import umap.plot as uplot
import umap
import torch
import seaborn as sns
import os



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
            if mtype == 'mlp':
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
                       metric='euclidean').fit(train_emb.numpy())

    return mapper.embedding_, train_labels
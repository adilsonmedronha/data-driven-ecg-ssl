import torch
import os

from models.Series2Vec import Series2Vec
from models.Series2Vec.S2V_training import *
from utils.utils import load_model
from models.model_factory import Model_factory
from Dataset import dataloader
from models.optimizers import get_optimizer, get_loss_module
from Dataset.dataloader import dataset_class
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from typing import Any, Optional
from models.heads.MLP import MLP
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

config = {
    "Model_Type": [
        "Series2Vec"
    ],
    "Norm": False,
    "Training_mode": "Pre_Training",
    "batch_size": 64,
    "console": False,
    "data_dir": "/home/adilson/Git/SSL-BRACIS-25/Series2Vec/Dataset/Benchmarks",
    "dataset": "Benchmarks",
    "dim_ff": 256,
    "dropout": 0.01,
    "emb_size": 16,
    "epochs": 600,
    "gpu": 0,
    "key_metric": "accuracy",
    "layers": 4,
    "lr": 0.001,
    "num_heads": 8,
    "output_dir": "Results/Pre_Training/Benchmarks/2025-03-29_07-38",
    "pred_dir": "Results/Pre_Training/Benchmarks/2025-03-29_07-38/predictions",
    "print_interval": 10,
    "rep_size": 320,
    "save_dir": "Results/Pre_Training/Benchmarks/2025-03-29_07-38/checkpoints",
    "seed": 1234,
    "tensorboard_dir": "Results/Pre_Training/Benchmarks/2025-03-29_07-38/tb_summaries",
    "val_interval": 2,
    "val_ratio": 0.2
}

problem = os.listdir(config['data_dir'])

config['problem'] = problem[1]

save_path = os.path.join(config['save_dir'], config['problem'] + '_pretrained_model_{}.pth'.format('last'))

DataWhereS2VwereTrained = dataloader.data_loader(config)
model = Model_factory(config, DataWhereS2VwereTrained)
optim_class = get_optimizer("RAdam")
config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
config['loss_module'] = get_loss_module()
model.to(config['gpu'])

SS_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])  # Loading the model
SS_Encoder.to(config['gpu'])

train = torch.load('Dataset/fragment/ecg-fragment_360hz/train.pt')
test = torch.load('Dataset/fragment/ecg-fragment_360hz/test.pt')
val = torch.load('Dataset/fragment/ecg-fragment_360hz/val.pt')

ytrain = train['labels']
ytest = test['labels']
yval = val['labels']

xtrain = train['samples']
xtest = test['samples']
xval = val['samples']

train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(xval, yval), batch_size=32, shuffle=True)


import torch
import wandb
from torch.utils.data import DataLoader


import wandb
import umap.plot as uplot
import umap

RUN_NAME = "ssl_retraining_and_mlp_as_head"
HEAD_TYPE = "mlp"
wandb.init()
wandb.run.name = RUN_NAME
wandb.run.save()
import matplotlib.pyplot as plt

from mine_utils import plot_umap, umap_embedding



def train(head_model, ssl_model, head_optimizer, ssl_optimizer, loss_module, train_loader, device, is_full_training=True):
    head_model.train() 
    ssl_model.train() if is_full_training else ssl_model.eval()
    epoch_loss = []
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        head_optimizer.zero_grad()
        ssl_optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_full_training):
            z_emb = ssl_model.linear_prob(x)
        
        y_hat = head_model(z_emb)
        loss = loss_module(y_hat, y)
        loss.backward()
        head_optimizer.step()
        
        epoch_loss.append(loss.item())
    
    avg_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_loss

def val(head_model, ssl_model, val_loader: DataLoader, loss_module, device='cuda'):
    head_model.eval()
    ssl_model.eval()
    val_loss = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            z_emb = ssl_model.linear_prob(x)
            y_hat = head_model(z_emb)
            loss = loss_module(y_hat, y)
            val_loss.append(loss.item())
    
    avg_val_loss = sum(val_loss) / len(val_loss)
    return avg_val_loss

import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

def test(head_model, ssl_model, test_loader, loss_module, device):
    head_model.eval()
    ssl_model.eval()
    test_loss = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            z_emb = ssl_model.linear_prob(x)
            y_hat = head_model(z_emb)
            loss = loss_module(y_hat, y)
            test_loss.append(loss.item())
            
            # Get predictions
            preds = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_test_loss = sum(test_loss) / len(test_loss)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = f1_score(all_labels, all_preds, average='macro')
    precision = f1_score(all_labels, all_preds, average='micro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    results = {
        "avg_loss": avg_test_loss,
        "accuracy": acc,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
        "confusion_matrix": conf_matrix
    }

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=range(6), yticklabels=range(6))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix.pdf")
    plt.close()

    return results

def run(run_idx, head_model, 
        ssl_model, 
        head_optimizer, 
        ssl_optimizer, loss_module, train_loader, val_loader, head_config, ssl_config, save_path, is_full_training=False):
    
    print(head_config.get('wandb_project', 'default_project'))
    config = {'head_model': head_config, 's2v_ssl_model': ssl_config}
    #wandb.init(project='ssl_finetuning', entity='adilson', config=config)
    #wandb.config.update(config)
    best_val_loss = float('inf')
    best_model = None
    device = head_config['gpu']
    EPOCHS = 2

    umap_ssl_emb_before, train_labels = umap_embedding(ssl_model, train_loader, device, mtype='ssl', name = 'UMAP_ssl_emb_before', save_path=save_path)
    fig_umap_before = plot_umap(umap_ssl_emb_before, 
                                train_labels, 
                                name = f"UMAP SSL BEFORE {RUN_NAME}",
                                save_path=os.path.join(save_path, f"UMAP_ssl_before_{RUN_NAME}.pdf"))
    for epoch in range(EPOCHS): 
        train_loss = train(head_model, ssl_model, head_optimizer, ssl_optimizer, loss_module, train_loader, device, is_full_training)
        val_loss = val(head_model, ssl_model, val_loader, loss_module, device)
        
        wandb.log({f'{run_idx}_Loss/train': train_loss, f'{run_idx}_Loss/val': val_loss, 'epoch': epoch})
        print(f'Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_head_model = head_model
    
    umap_ssl_emb_after_finetuning, _  = umap_embedding(ssl_model, train_loader, device, mtype='ssl', name = 'UMAP_ssl_emb_after_finetuning',save_path=save_path)
    fig_umap_after = plot_umap(umap_ssl_emb_after_finetuning, 
                               train_labels, 
                               name = f"UMAP SSL AFTER {RUN_NAME}",
                               save_path = os.path.join(save_path, f"UMAP_ssl_after_{RUN_NAME}.pdf"))
    
    umap_head_emb_after_finetuning, _ = umap_embedding(ssl_model, train_loader, device, head_model, mtype=HEAD_TYPE, name = 'UMAP_head_emb_after_finetuning', save_path=save_path)
    fig_umap_mlp_after = plot_umap(umap_head_emb_after_finetuning, 
                                   train_labels, 
                                   name = f"UMAP {HEAD_TYPE} BEFORE {RUN_NAME}",
                                   save_path = os.path.join(save_path, f"UMAP_head_after_{RUN_NAME}.pdf"))
    
    wandb.log({"umap_before": fig_umap_before, "umap_after": fig_umap_after, "umap_mlp_after": fig_umap_mlp_after})

    return {
        "val_loss": best_val_loss,
        "head_model": best_head_model,
        "ssl_model": ssl_model,
    }


if __name__ == '__main__':
    in_dim = 640 
    mlp = MLP(640, 128, 6)
    device = config['gpu']
    mlp.to(device)
    adam_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    N_EXPERIMENTS = 2

    initial_timestamp = datetime.now()
    output_dir = os.path.join("Results", "Pos_training", initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(columns=["model", "dataset", "exp", "avg loss", "acc", "f1", "recall", "precision"])
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, mode='a', header=True, index=False)

    wandb.init(project='ssl_finetuning', entity='adilson', config=config)
    wandb.config.update({'head_model': config, 's2v_ssl_model': config})
    models = []
    for run_idx in range(N_EXPERIMENTS):
        train_results = run(run_idx, head_model = mlp, 
            ssl_model = SS_Encoder, 
            head_optimizer = adam_optimizer, 
            ssl_optimizer= config['optimizer'],
            loss_module = criterion,
            train_loader = train_loader, 
            val_loader = val_loader, 
            head_config = config,
            ssl_config = config,
            save_path = output_dir,
            is_full_training = True)
        models.append(train_results)

        # TODO nome do experiment no wandb
        results = test(head_model = mlp, 
                            ssl_model = SS_Encoder, 
                            test_loader = test_loader, 
                            loss_module = criterion,
                            device = device)
        
        df = pd.DataFrame({
            "model": [RUN_NAME],
            "dataset": [config['problem']],
            "exp": [run_idx],
            "avg loss": [results["avg_loss"]],
            "acc": [results['accuracy']],
            "f1": [results['f1_score']],
            "recall": [results['recall']],
            "precision": [results['precision']],
        })

        df.to_csv(csv_path, mode='a', header = not pd.io.common.file_exists(csv_path), index=False)


    best_ssl_and_head = min(models, key=lambda x: x['val_loss'])
    torch.save(best_ssl_and_head['head_model'].state_dict(), os.path.join(output_dir, f'{RUN_NAME}_best_head_model.pth'))
    torch.save(best_ssl_and_head['ssl_model'].state_dict(),  os.path.join(output_dir, f'{RUN_NAME}_ssl_model.pth'))
    wandb.finish()
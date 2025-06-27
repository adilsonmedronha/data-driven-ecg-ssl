import os
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from mine_utils import umap_embedding, plot_umap


def metrics_name(task):
    if task == 'classification':
        return ['acc', 'f1', 'recall', 'precision']
    elif task == 'regression':
        return ['mse', 'r2_score', 'rmse', 'mae']
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks are 'classification' and 'regression'.")


def train(head_model, ssl_model, head_optimizer, ssl_optimizer, loss_module, train_loader, device, is_finetuning=True):
    # Placeholder for training logic
    head_model.train() 
    ssl_model.train() if is_finetuning else ssl_model.eval()
    epoch_loss = []
    is_finetuning = bool(is_finetuning)
    for (x, y) in train_loader:
        x, y = x.to(device), y.to(device)
        head_optimizer.zero_grad()
        if is_finetuning:
            ssl_optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_finetuning):
            z_emb = ssl_model.linear_prob(x)
        y_hat = head_model(z_emb)
        loss = loss_module(y_hat, y)
        loss.backward()
        head_optimizer.step()

        if is_finetuning:
            ssl_optimizer.step()
            if ssl_model._get_name() == "TS2Vec":
                ssl_model.swa_update_params()

        epoch_loss.append(loss.item())
    
    avg_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_loss

def val(head_model, ssl_model, val_loader, loss_module, device='cuda'):
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
    return avg_val_loss, val_loss


def test(head_model, ssl_model, test_loader, loss_module, task, device, save_path, run_name):
    # Placeholder for test logic
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
            
            if task == 'classification': # return predicted labels
                preds = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
            else:
                preds = y_hat

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    results = {}
    if task == 'classification':
        results = _classification_metrics(all_labels, all_preds, save_path, run_name)
    elif task == 'regression':
        results = _regression_metrics(all_labels, all_preds, save_path, run_name)
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks are 'classification' and 'regression'.")

    results['avg_loss'] = sum(test_loss) / len(test_loss)

    return results


def _classification_metrics(all_labels, all_preds, save_path, run_name):
    acc = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
    recall = metrics.f1_score(all_labels, all_preds, average='macro')
    precision = metrics.f1_score(all_labels, all_preds, average='micro')
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)

    results = {
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "confusion_matrix": conf_matrix
    }

    # Save classification metrics to a file
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=range(6), yticklabels=range(6))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_path, f"confusion_matrix_{run_name}.pdf"))
    plt.close()

    return results


def _regression_metrics(all_labels, all_preds, save_path, run_name):
    mse = metrics.mean_squared_error(all_labels, all_preds)
    r2 = metrics.r2_score(all_labels, all_preds)
    rmse = metrics.root_mean_squared_error(all_labels, all_preds)
    mae = metrics.mean_absolute_error(all_labels, all_preds)

    results = {
        "mse": mse,
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae
    }

    # Save regression metrics to a file
    plt.figure(figsize=(10, 8))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression Predictions vs True Values')
    plt.savefig(os.path.join(save_path, f"regression_plot_{run_name}.pdf"))
    plt.close()

    return results

def plot_train_and_val_loss(train_loss, val_loss, save_path, filename, title):
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(save_path, f"train_val_loss_{filename}.pdf"))
    plt.close()

def plot_embedding_with_umap(ssl_model, train_loader, device, save_path, filename, title, head_model=None):

    umap_embed, labels = umap_embedding(model = ssl_model, 
                                        dataloader = train_loader,
                                        device = device, 
                                        head_model = head_model,
                                        mtype = 'ssl' if head_model is None else head_model._get_name(),
                                        name = filename, 
                                        save_path = save_path)
    
    fig_umap = plot_umap(embedding = umap_embed, 
                                train_labels = labels, 
                                name = title,
                                save_path = os.path.join(save_path, f"{filename}.pdf"))
    
    return fig_umap#, umap_embed, labels
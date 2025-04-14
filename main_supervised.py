import torch
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models.supervised.Hinception import Hinception, HinceptionTime
from mine_utils import load_fragment_dataset
from torchsummary import summary
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse


import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths.*")
from numba.core.errors import NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", category=Warning)
import seaborn as sns
from mine_utils import plot_umap, tsne_embedding,  umap_embedding, set_seed, get_parser, load_fragment_dataset
import pandas as pd
from datetime import datetime

import warnings
from numba.core.errors import NumbaWarning
import wandb

warnings.filterwarnings(
    "ignore",
    category=NumbaWarning,
    message=".*Compilation requested for previously compiled argument types.*"
)


def val(model, val_loader, loss_module, device='cuda'):
    model.eval()
    val_loss = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_module(output, y)
            val_loss.append(loss.item())

            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = f1_score(all_labels, all_preds, average='macro')
    precision = f1_score(all_labels, all_preds, average='micro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    avg_val_loss = sum(val_loss) / len(val_loss)

    return  {
        "epoch_val_loss": avg_val_loss,
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "confusion_matrix": conf_matrix
    }
    


def train(model, train_loader, optimizer, scheduler, criterion, device='cuda') -> torch.Tensor:
    model.to(device)
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []
    for (x, y) in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = output.argmax(dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    scheduler.step(running_loss / len(train_loader))
    return model, running_loss/len(train_loader), acc


def test(model,  test_loader, device, save_path, RUN_NAME='teste'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (x, y) in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=1).cpu().numpy()
            preds = torch.argmax(F.softmax(output, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = f1_score(all_labels, all_preds, average='macro')
    precision = f1_score(all_labels, all_preds, average='micro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    results = {
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
    plt.savefig(os.path.join(save_path, f"confusion_matrix_{RUN_NAME}.pdf"))
    plt.close()
    return results

def run(run_idx, num_epochs, train_loader, val_loader, model, optimizer, scheduler, criterion, device='cuda', path=''): 
    min_val_loss = float('inf')
    val_loss_epochs = []
    for epoch in (range(num_epochs)):
        model, running_loss, acc = train(model, train_loader, optimizer, scheduler, criterion, device)
        val_results              = val(model, val_loader, criterion, device)
        val_loss = val_results['epoch_val_loss']
        print(f"Run {run_idx+1}/Epoch {epoch+1}, Train Loss: {running_loss}, Val Loss: {val_loss}, Acc Train: {acc}, Acc Val {val_results['acc']}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"{path}/best_model_run_{run_idx}.pth")
        val_loss_epochs.append(val_loss)

        wandb.log({
            f"train_loss/run_{run_idx}": running_loss,
            f"val_loss/run_{run_idx}": val_loss,
            f"val_acc/run_{run_idx}": val_results['acc'],
            f"step/run_{run_idx}": epoch  
        })
        
    torch.save(torch.tensor(val_loss_epochs), f"{path}/validation_loss.pth")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument("--nro_models", type=int, default=5, help="Number of models to train in parallel")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--runs", type=int, default=3, help="Number of experiment repetitions")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--sequence_length", type=int, default=720, help="Input sequence length")
    parser.add_argument("--output_dir", type=str, default="Results/Baselines/", help="Directory to save results")
    parser.add_argument("--RUN_NAME", type=str)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    # python main_supervised.py --batch_size 64 --nro_models 5 --runs 5 --seed 123 --RUN_NAME "Hinception and HinceptionTime"

    models_names = [f'Hinception_{i+1}' for i in range(args.nro_models)]
    initial_timestamp = datetime.now()
    run_output_dir = os.path.join(args.output_dir, initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(run_output_dir, exist_ok=True)
    device = 'cuda'

    seeds = [args.seed + i for i in range(args.runs)]
    for idx in range(args.nro_models):
        set_seed(seeds[idx])
        train_loader, val_loader, test_loader = load_fragment_dataset(batch_size=args.batch_size)
        curr_model = Hinception(sequence_length=args.sequence_length, in_channels=1, num_classes=6)
        curr_opt   = torch.optim.Adam(curr_model.parameters())
        curr_sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(curr_opt, mode='min', factor=0.1, patience=10, min_lr=0)
        curr_loss  = nn.CrossEntropyLoss()
        model_name = models_names[idx]
        curr_run_output_dir = os.path.join(run_output_dir, model_name)
        os.makedirs(curr_run_output_dir, exist_ok=True)
        wandb.init(project='ssl_pretrained_on_multidomain_dataset', entity='adilson', name =f'{args.RUN_NAME} Model {idx + 1}')
        wandb.run.save()

        csv_path = os.path.join(curr_run_output_dir, "results.csv")
        if not os.path.isfile(csv_path):
            df = pd.DataFrame(columns=["model", "dataset", "exp", "acc", "f1", "recall", "precision"])
            df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f'\nTraining Hinception {idx+1}\n')

        for run_idx in range(args.runs):
            run(run_idx,
                args.num_epochs,
                train_loader,
                val_loader,
                curr_model,
                curr_opt,
                curr_sch,
                curr_loss,
                device=device,
                path=curr_run_output_dir)

            results = test(curr_model, test_loader, device, curr_run_output_dir)

            df = pd.DataFrame({
                "model": [model_name],
                "dataset": ['fragment'],
                "exp": [run_idx],
                "acc": [results['accuracy']],
                "f1": [results['f1_score']],
                "recall": [results['recall']],
                "precision": [results['precision']],
            })
            df.to_csv(csv_path, mode='a', header=False, index=False)
        wandb.finish()

if __name__ == "__main__":
    main()
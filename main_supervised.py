import os
import torch
from torch import nn
from torch.nn import functional as F

import wandb
from models.heads.supervised.FCN import FCN
from models.heads.supervised.MLP import MLP
from models.heads.supervised.Hinception import HinceptionTime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import json

from numba.core.errors import NumbaWarning
import seaborn as sns
from mine_utils import set_seed, get_parser, load_fragment_dataset
import pandas as pd
from datetime import datetime


import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths.*")
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings(
    "ignore",
    category=NumbaWarning,
    message=".*Compilation requested for previously compiled argument types.*"
)
    

def test(model, test_loader, device, save_path, RUN_NAME='teste'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (x, y) in test_loader:
            x, y = x.to(device), y.to(device)
            if isinstance(model, HinceptionTime):
                preds = model(x)
            else:
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wdecay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--runs", type=int, default=5, help="Number of experiment repetitions")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--sequence_length", type=int, default=720, help="Input sequence length")
    parser.add_argument("--output_dir", type=str, default="Results/Supervised/", help="Directory to save results")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    initial_timestamp = datetime.now()
    run_output_dir = os.path.join(args.output_dir, initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(run_output_dir, exist_ok=True)
    device = 'cuda'
    seeds = [args.seed + i for i in range(args.runs)]
    model_params = {  HinceptionTime: {'sequence_length': 720, 'in_channels': 1, 'num_classes': 6},
                      FCN: {'in_dim': 1, 'num_classes': 6},
                      MLP: {'in_dim': 720, 'hidden_dim': 500, 'output_size': 6}  }
    
    configs = {
        'lr': args.lr,
        'weight_decay': args.wdecay,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'sched_params': {'mode': 'min', 'factor': 0.1, 'patience': 10, 'min_lr': 0},
        'loss': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam,
    }

    print(configs)

    for idx, model in enumerate(model_params):
        set_seed(seeds[idx])
        train_loader, val_loader, test_loader = load_fragment_dataset(batch_size=args.batch_size)
        curr_model = model(**model_params[model], configs=configs)
        model_name = f'{curr_model.__class__.__name__}' if isinstance(curr_model, HinceptionTime) else f'{curr_model.__class__.__name__}'  
        curr_run_output_dir = os.path.join(run_output_dir, model_name)
        os.makedirs(curr_run_output_dir, exist_ok=True)
        wandb.init(project='ssl_pretrained_on_multidomain_dataset', entity='labic-icmc', name =f'{model_name}')
        wandb.run.save()

        csv_path = os.path.join(curr_run_output_dir, "results.csv")
        if not os.path.isfile(csv_path):
            df = pd.DataFrame(columns=["model", "dataset", "exp", "acc", "f1", "recall", "precision"])
            df.to_csv(csv_path, mode='w', header=True, index=False)

        for run_idx in range(args.runs):
            print(f' TRAINING {curr_model.__class__.__name__}')
            curr_model.run(train_loader, val_loader, args.num_epochs, device, run_idx=run_idx, path=curr_run_output_dir)
            curr_model.to(device)
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

        configs_bkp = configs.copy()
        del configs_bkp['sched_params']
        del configs_bkp['loss']
        del configs_bkp['optimizer']
        del configs_bkp['scheduler']

        full_config = {
            'configs': configs_bkp,
            'dataset': 'fragment',
            'runs': args.runs,
            'seed': args.seed,
            'sequence_length': args.sequence_length,
            'output_dir': run_output_dir
        }
        
        full_config_path = os.path.join(curr_run_output_dir, "full_config.json")
        with open(full_config_path, "w") as f:
            json.dump(full_config, f, indent=4)

if __name__ == "__main__":
    main()
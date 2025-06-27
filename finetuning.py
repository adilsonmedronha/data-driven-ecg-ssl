import torch
import os
import json
import wandb
import pandas as pd
from datetime import datetime
from models.Series2Vec.S2V_training import *
from utils.utils import load_model
from models.model_factory import Model_factory
from Dataset import dataloader
from models.optimizers import get_optimizer
from mine_utils import select_head, set_seed, get_parser, load_task_dataset
from utils import task_trainer

import torch
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")


def save_json(dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(dict, f, indent=4)
    print(f"Configuration saved to {file_path}")


def run(run_idx, epochs, head_model, ssl_model, head_optimizer, 
        ssl_optimizer, loss_module, train_loader, val_loader, save_path, device='cuda', is_finetuning=False):
    print(head_config.get('wandb_project', 'default_project'))
    best_avg_val_loss = float('inf')

    fig_umap_before = task_trainer.plot_embedding_with_umap(
        ssl_model=ssl_model,
        train_loader=train_loader,
        device=device,
        save_path=save_path,
        filename=f"UMAP_encoder_before_{RUN_NAME}",
        title=f"UMAP SSL BEFORE {RUN_NAME}",
    )
    train_losses = []
    val_losses = []
    for epoch in range(epochs): 
        train_loss = task_trainer.train(head_model,
                           ssl_model, 
                           head_optimizer, 
                           ssl_optimizer, loss_module, train_loader, device, is_finetuning)
        avg_val_loss, val_loss = task_trainer.val(head_model, ssl_model, val_loader, loss_module, device)
        train_losses.append(train_loss)
        val_losses.append(avg_val_loss)
        wandb.log({
            f"train_loss/run_{run_idx}": train_loss,
            f"avg_val_loss/run_{run_idx}": avg_val_loss,
            f"val_loss/run_{run_idx}": val_loss,
            f"step/run_{run_idx}": epoch  
        })
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            best_head_model = head_model
            best_ssl_model = ssl_model
            best_epoch = epoch
    
    fig_umap_after = task_trainer.plot_embedding_with_umap(
        ssl_model=best_ssl_model,
        train_loader=train_loader,
        device=device,
        save_path=save_path,
        filename=f"UMAP_encoder_after_{RUN_NAME}",
        title=f"UMAP SSL AFTER {RUN_NAME}",
    )

    fig_umap_mlp_after = task_trainer.plot_embedding_with_umap(
        ssl_model=best_ssl_model,
        head_model=best_head_model,
        train_loader=train_loader,
        device=device,
        save_path=save_path,
        filename=f"UMAP_head_after_{RUN_NAME}",
        title=f"UMAP {best_head_model._get_name()} BEFORE {RUN_NAME}",
    )
    
    print(f"Best epoch of this run: {best_epoch}")
    wandb.log({"umap_before": fig_umap_before, 
               "umap_after": fig_umap_after, 
               "umap_mlp_after": fig_umap_mlp_after})
    

    losses_json = { 'train_loss': train_losses, 'val_loss': val_losses }
    save_json(losses_json, os.path.join(save_path, f"train_val_loss_{epochs}_eps_run_{run_idx}.json"))

    task_trainer.plot_train_and_val_loss(
        train_loss=train_losses,
        val_loss=val_losses,
        save_path=save_path,
        filename=f"train_val_loss_{epochs}_epochs_{run_idx}",
        title=f"Train and Val Loss {epochs} epochs {run_idx}",
    )
    return {
        "best_avg_val_loss": best_avg_val_loss,
        "best_head_model": best_head_model,
        "best_ssl_model": best_ssl_model,
        "best_epoch": best_epoch,
    }


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    with open(args.encoder_configuration_file, 'r') as f:
        encoder_config = json.load(f)

    with open(args.head_configuration_file, 'r') as f:
        head_config = json.load(f)

    device = head_config['gpu']
    N_EXPERIMENTS = args.runs

    # ----------------------------------------------- Configure the Wandb -----------------------------------------------
    RUN_NAME = args.description
    wandb.init(project='ssl_pretrained_on_multidomain_dataset',
           entity='labic-icmc',
           name= RUN_NAME)
    wandb.config.update({'head_model': head_config, 'tstcc_ssl_model': encoder_config})
    wandb.run.save()
    
    # ------------------------------------------------ Create output dir ------------------------------------------------
    initial_timestamp = datetime.now()
    output_dir = os.path.join("Results", "Pos_training", args.folder_name, initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(columns=["model", "dataset", "exp", "avg loss"]+task_trainer.metrics_name(head_config['task']))
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, mode='a', header=True, index=False)
    
    # ----------------------------------------------- Run the experiment -----------------------------------------------
    # Load Target dataset
    train_loader, val_loader, test_loader = load_task_dataset(head_config['batch_size'], head_config['dataset'])

    path2load_encoder = os.path.join(args.encoder_checkpoint_path)
    DataWhereS2VwereTrained = dataloader.data_loader(encoder_config)

    models = []
    seeds = [args.seed + i for i in range(N_EXPERIMENTS)]
    for run_idx in range(N_EXPERIMENTS):
        set_seed(seeds[run_idx])
        # Load SSL model -------------------------------------------------
        model = Model_factory(encoder_config, DataWhereS2VwereTrained)
        encoder_config['optimizer'] = get_optimizer("RAdam", model, encoder_config)
        SS_Encoder, optimizer, start_epoch = load_model(model, path2load_encoder, encoder_config['Model_Type'], encoder_config['optimizer'])
        SS_Encoder.to(encoder_config['gpu'])

        # Load SSL model -------------------------------------------------
        head_model, adam_optimizer, criterion = select_head(args.model_name, head_config)

        train_results = run(run_idx, 
                            epochs= args.epochs,
                            head_model = head_model, 
                            ssl_model = SS_Encoder, 
                            head_optimizer = adam_optimizer, 
                            ssl_optimizer= encoder_config['optimizer'],
                            loss_module = criterion,
                            train_loader = train_loader, 
                            val_loader = val_loader, 
                            save_path = output_dir,
                            is_finetuning = args.is_finetuning)
        
        models.append(train_results)
        results = task_trainer.test(head_model = head_model,
                       ssl_model = SS_Encoder, 
                       test_loader = test_loader, 
                       loss_module = criterion,
                       task = head_config['task'],
                       device = device,
                       save_path = output_dir, run_name=RUN_NAME)
        
        df = pd.DataFrame({
            "model": [RUN_NAME],
            "dataset": [head_config['dataset']],
            "exp": [run_idx],
            **{k: [v] for k, v in results.items() if k != 'confusion_matrix'},
        })
        df.to_csv(csv_path, mode='a', header = not pd.io.common.file_exists(csv_path), index=False)

    # save the best model among the five runs
    best_ssl_and_head = min(models, key=lambda x: x['best_avg_val_loss'])
    #print(best_ssl_and_head['best_ssl_model'].state_dict().keys(), len(best_ssl_and_head['best_ssl_model'].state_dict().keys()))

    torch.save(best_ssl_and_head['best_head_model'].state_dict(), os.path.join(output_dir, f'{RUN_NAME}_best_head_model_across_all_loss_validation.pth'))
    torch.save(best_ssl_and_head['best_ssl_model'].state_dict(), os.path.join(output_dir, f'{RUN_NAME}_best_ssl_model_weights.pth'))

    #print(len(torch.load(os.path.join(output_dir, f'{RUN_NAME}_best_ssl_model_weights.pth')).keys()))

    # save the config files 
    save_json(head_config, os.path.join(output_dir, f'{RUN_NAME}_head_config.json'))
    save_json(args.__dict__, os.path.join(output_dir, f'{RUN_NAME}_args.json'))

    wandb.finish()



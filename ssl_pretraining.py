import os
import json
import logging
import argparse
import numpy as np
from numba import config as numba_config

from Dataset import dataloader
from models.Series2Vec.runner import pre_training as Series2Vec_pre_training
from models.TS2Vec.runner import pre_training as TS2Vec_pre_training
from models.TSTCC.runner import pre_training as TSTCC_pre_training
from mine_utils import set_seed, load_task_dataset

# Set the Numba configuration to enable PYNVJITLINK
numba_config.CUDA_ENABLE_PYNVJITLINK = 1

logger = logging.getLogger(__name__)


def including_task_dataset(config, task_dataset, Data):
    train_loader, val_loader, _ = load_task_dataset(config['batch_size'], task_dataset)

    logger.info(f"Including task dataset '{args.task_dataset}' during the pretraining")
    Data['train_data'] = np.concat((Data['train_data'], train_loader.dataset.tensors[0].numpy()), axis=0)
    Data['train_label'] = np.concat((Data['train_label'], train_loader.dataset.tensors[1].numpy()), axis=0)

    Data['test_data'] = np.concat((Data['test_data'], val_loader.dataset.tensors[0].numpy()), axis=0)
    Data['test_label'] = np.concat((Data['test_label'], val_loader.dataset.tensors[1].numpy()), axis=0)

    logger.info(f"  - Inserted {train_loader.dataset.tensors[0].shape[0]} training samples from task dataset")
    logger.info(f"  - Inserted {val_loader.dataset.tensors[0].shape[0]} testing samples from task dataset")

    return Data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the head model')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from the last checkpoint', action=argparse.BooleanOptionalAction)
    parser.add_argument('--datasets', type=str, default=None, help='Path to the data directory', nargs='*')
    parser.add_argument('--task_dataset', type=str, default=None, help='Path to the task directory')
    args = parser.parse_args()

    # Read the configuration file
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    config['epochs'] = args.epochs
    set_seed(config['seed'])

    config['resume'] = args.resume
    if args.resume:
        config['old-problem'] = config['problem']
        config['old-config'] = args.config_file

    dataset_list = os.listdir(config['data_dir']) if args.datasets is None else args.datasets
    # ------------------------------------------- Pretrain Series2Vec -------------------------------------------
    if config['Model_Type'] == 'Series2Vec':
        for problem in dataset_list:
            config['problem'] = problem
            print(problem)
            
            Data = dataloader.data_loader(config)

            # include the task dataset during the pretraining
            if args.task_dataset is not None:
                Data = including_task_dataset(config, args.task_dataset, Data)
                config['problem'] = f"{config['problem']}+{args.task_dataset}"
                
            Series2Vec_pre_training(config, Data, resume_train=args.resume)
            
            # Save the configs to a file ---------------------------------------------------
            config_path = os.path.join(config['save_dir'], config['problem']+'_TS2Vec_config.json')

            final_dict = {k: v for k, v in config.items() if k not in ['optimizer', 'loss_module']}
            json.dump(final_dict, open(config_path, 'w'))
            print(f"Config saved to {config_path}")


    # ------------------------------------------- Pretrain TS2Vec -------------------------------------------
    elif config['Model_Type'] == 'TS2Vec':
        for problem in dataset_list:
            
            config['problem'] = problem
            print(problem)
            
            Data = dataloader.data_loader(config)

            # include the task dataset during the pretraining
            if args.task_dataset is not None:
                Data = including_task_dataset(config, args.task_dataset, Data)
                config['problem'] = f"{config['problem']}+{args.task_dataset}"

            updated_config, logs = TS2Vec_pre_training(config, Data, resume_train=args.resume)

            # Save the configs to a file ---------------------------------------------------
            logs_path = os.path.join(config['save_dir'], config['problem']+'_TS2Vec_logs.json')
            config_path = os.path.join(config['save_dir'], config['problem']+'_TS2Vec_config.json')

            json.dump(logs, open(logs_path, 'w'))
            print(f"Logs saved to {logs_path}")

            json.dump(updated_config, open(config_path, 'w'))
            print(f"Config saved to {config_path}")


    # ------------------------------------------- Pretrain TSTCC -------------------------------------------
    elif config['Model_Type'] == 'TSTCC':
        for problem in dataset_list:

            config['problem'] = problem
            print(problem)
            
            Data = dataloader.data_loader(config)

            # include the task dataset during the pretraining
            if args.task_dataset is not None:
                Data = including_task_dataset(config, args.task_dataset, Data)
                config['problem'] = f"{config['problem']}+{args.task_dataset}"

            updated_config, logs = TSTCC_pre_training(config, Data, resume_train=args.resume)

            # Save the logs to a file 
            logs_path = os.path.join(config['save_dir'], config['problem']+'_TSTCC_logs.json')
            json.dump(logs, open(logs_path, 'w'))
            print(f"Logs saved to {logs_path}")

            # Save the configs to a file 
            config_path = os.path.join(config['save_dir'], config['problem']+'_TSTCC_config.json')
            json.dump(updated_config, open(config_path, 'w'))
            print(f"Config saved to {config_path}")


    print('Pretraining completed.')

import os
import json
import argparse
from numba import config as numba_config

from Dataset import dataloader
from models.Series2Vec.runner import pre_training as Series2Vec_pre_training
from models.TS2Vec.runner import pre_training as TS2Vec_pre_training
from models.TSTCC.runner import pre_training as TSTCC_pre_training


# Set the Numba configuration to enable PYNVJITLINK
numba_config.CUDA_ENABLE_PYNVJITLINK = 1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the head model')
    args = parser.parse_args()

    # Read the configuration file
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    config['epochs'] = args.epochs

    # ------------------------------------------- Pretrain Series2Vec -------------------------------------------
    if config['Model_Type'] == 'Series2Vec':
        for problem in os.listdir(config['data_dir']):
            config['problem'] = problem
            print(problem)
            
            Data = dataloader.data_loader(config)
            Series2Vec_pre_training(config, Data)


    # ------------------------------------------- Pretrain TS2Vec -------------------------------------------
    elif config['Model_Type'] == 'TS2Vec':
        for problem in os.listdir(config['data_dir']):
            config['problem'] = problem
            print(problem)
            
            Data = dataloader.data_loader(config)
            updated_config, logs = TS2Vec_pre_training(config, Data)

            # Save the configs to a file ---------------------------------------------------
            logs_path = os.path.join(config['save_dir'], config['problem']+'_TS2Vec_logs.json')
            config_path = os.path.join(config['save_dir'], config['problem']+'_TS2Vec_config.json')

            json.dump(logs, open(logs_path, 'w'))
            print(f"Logs saved to {logs_path}")

            json.dump(updated_config, open(config_path, 'w'))
            print(f"Config saved to {config_path}")


    # ------------------------------------------- Pretrain TSTCC -------------------------------------------
    elif config['Model_Type'] == 'TSTCC':
        for problem in os.listdir(config['data_dir']):
            config['problem'] = problem
            
            Data = dataloader.data_loader(config)
            updated_config, logs = TSTCC_pre_training(config, Data)

            # Save the logs to a file 
            logs_path = os.path.join(config['save_dir'], config['problem']+'_TSTCC_logs.json')
            json.dump(logs, open(logs_path, 'w'))
            print(f"Logs saved to {logs_path}")

            # Save the configs to a file 
            config_path = os.path.join(config['save_dir'], config['problem']+'_TSTCC_config.json')
            json.dump(updated_config, open(config_path, 'w'))
            print(f"Config saved to {config_path}")


    print('Pretraining completed.')
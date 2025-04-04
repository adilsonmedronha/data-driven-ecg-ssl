import os
from utils import args
from Dataset import dataloader
from models.runner import supervised, pre_training, linear_probing
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

if __name__ == '__main__':
    config = args.Initialization(args)

for problem in os.listdir(config['data_dir']):
    config['problem'] = problem
    print(problem)
    Data = dataloader.data_loader(config)
    best_aggr_metrics_test, all_metrics = pre_training(config, Data)

    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)

    with open(os.path.join(config['output_dir'], config['problem']+'_output.txt'), 'w') as file:
        for k, v in all_metrics.items():
            file.write(f'{k}: {v}\n')
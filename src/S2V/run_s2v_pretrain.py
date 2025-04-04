
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

    if config['Training_mode'] == 'Pre_Training':
        if config['Model_Type'][0] == 'Series2Vec':
            best_aggr_metrics_test, all_metrics = pre_training(config, Data)
    elif config['Training_mode'] == 'Linear_Probing':
        best_aggr_metrics_test, all_metrics = linear_probing(config, Data)
    elif config['Training_mode'] == 'Supervised':
        best_aggr_metrics_test, all_metrics = supervised(config, Data)

    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)

    with open(os.path.join(config['output_dir'], config['problem']+'_output.txt'), 'w') as file:
        for k, v in all_metrics.items():
            file.write(f'{k}: {v}\n')


def pre_training(config, Data):
    model = Model_factory(config, Data)
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    model.to(config['device'])

    # --------------------------------- Load Data ---------------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    x_train,y_train, _ = next(iter(train_loader))
    x_test,y_test, _ = next(iter(test_loader))
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    
    # --------------------------------- Self Superviseed Training ------------------------------------------------------
    SS_trainer = S2V_SS_Trainer(model, train_loader, test_loader, config, print_conf_mat=False)
    save_path = os.path.join(config['save_dir'], config['problem'] + '_pretrained_model_{}.pth'.format('last'))
    SS_train_runner(config, model, SS_trainer, save_path)
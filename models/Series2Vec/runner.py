import os
from datetime import datetime
from models.model_factory import Model_factory
from models.optimizers import get_optimizer, get_loss_module
from torch.utils.data import DataLoader
from Dataset.dataloader import dataset_class
from .S2V_training import *

from utils.utils import load_model


import logging

logger = logging.getLogger('__main__')


def choose_trainer(model, train_loader, test_loader, config, conf_mat, type):
    if config['Model_Type'] == 'Series2Vec':
        S_trainer = S2V_S_Trainer(model, train_loader, test_loader, config, print_conf_mat=conf_mat)
    return S_trainer


def pre_training(config, Data, resume_train=False):
    logger.info("Creating Distance based Self Supervised model ...")
    model = Model_factory(config, Data)
    config['optimizer'] = get_optimizer("RAdam", model, config)
    config['loss_module'] = get_loss_module()
    model.to(config['gpu'])

    # load the checkpoint if resume training ---------------------------------------------
    if resume_train:
        logger.info("Resuming the training ...")
        checkpoint_path = f"{config['save_dir']}/{config['old-problem']}_pretrained_model_last.pth"
        checkpoint = torch.load(checkpoint_path)
        
        logger.info(f"Loading the checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint['state_dict'])

        # update the save_dir to save the resumed training
        path_levels = config['save_dir'].split('/')
        path_levels[-2] = datetime.now().strftime("%Y-%m-%d_%H-%M")
        config['save_dir'] = '/'.join(path_levels)
        logger.info(f"Saving the resumed training in {config['save_dir']}")


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

    # create the dir to save the checkpoints
    if not os.path.isdir(config['save_dir']):
        os.makedirs(config['save_dir'])

    SS_train_runner(config, model, SS_trainer, save_path)

    return



def linear_probing():
    return


def supervised(config, Data):
    model = Model_factory(config, Data)
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    model.to(config['gpu'])

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = choose_trainer(model, train_loader, None, config, False, 'S')
    S_val_evaluator = choose_trainer(model, val_loader, None, config, False, 'S')
    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))

    Strain_runner(config, model, S_trainer, S_val_evaluator, save_path)
    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(config['gpu'])

    best_test_evaluator = choose_trainer(best_model, test_loader, None, config, True, 'S')
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics

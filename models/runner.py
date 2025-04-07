import os
from models.model_factory import Model_factory
from models.optimizers import get_optimizer, get_loss_module
from torch.utils.data import DataLoader
from Dataset.dataloader import dataset_class
from models.Series2Vec.S2V_training import *


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix

from utils.utils import load_model


import logging

logger = logging.getLogger('__main__')


def choose_trainer(model, train_loader, test_loader, config, conf_mat, type):
    if config['Model_Type'][0] == 'Series2Vec':
        S_trainer = S2V_S_Trainer(model, train_loader, test_loader, config, print_conf_mat=conf_mat)
    return S_trainer


def pre_training(config, Data):
    logger.info("Creating Distance based Self Supervised model ...")
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

    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    train_repr, train_labels = S2V_make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = S2V_make_representation(SS_Encoder, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    LP_acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', LP_acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)
    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    logger.info('\n\nStarting Fine_Tuning...')
    S_trainer = S2V_S_Trainer(SS_Encoder, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = S2V_S_Trainer(SS_Encoder, test_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_finetuned_model_{}.pth'.format('last'))
    Strain_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(SS_Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    train_repr, train_labels = S2V_make_representation(best_Encoder, train_loader)
    test_repr, test_labels = S2V_make_representation(best_Encoder, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)

    best_test_evaluator = S2V_S_Trainer(best_Encoder, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    all_metrics['LGR_ACC'] = acc_test
    all_metrics['LP_LGR_ACC'] = LP_acc_test
    return best_aggr_metrics_test, all_metrics



def linear_probing():
    return


def supervised(config, Data):
    model = Model_factory(config, Data)
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    model.to(config['device'])

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
    best_model.to(config['device'])

    best_test_evaluator = choose_trainer(best_model, test_loader, None, config, True, 'S')
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics

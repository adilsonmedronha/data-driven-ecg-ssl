import torch
import numpy as np
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader

from Dataset.dataloader import dataset_class
from .ts2vec import TS2Vec


import logging

logger = logging.getLogger('__main__')


def pre_training(config, data, resume_train = False):
    '''Pretrain the TS2Vec feature extractor model in the pretext task.

    Args:
        data (dict): Train dataset.
        config (dict): Trainig configuration.
        resume_train (bool): Resume training from a checkpoint. Default = False.

    Returns:
        pretrained TS2Vec params (dict), training logs (dict)
    '''
    logger.info("Creating the TS2Vec Self-supervised model ...")

    # re-arrange the dimensions to ts2vec pattern (n_instance, n_timestamps, n_features)
    data['train_data'] = np.swapaxes(data['train_data'], 1, 2)
    data['test_data'] = np.swapaxes(data['test_data'], 1, 2)

    logger.info(f"train shape {data['train_data'].shape} / test shape {data['test_data'].shape}")

    train_dataset = dataset_class(data['train_data'], data['train_label'], config)
    test_dataset = dataset_class(data['test_data'], data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # instanciate the model
    ts2vec = TS2Vec(device=config['gpu'], **config['model_args'])
    ts2vec_optim = optim.AdamW(ts2vec._net.parameters(), **config['optim_args'])


    # load the checkpoint if resume training ---------------------------------------------
    if resume_train:
        logger.info("Resuming the training ...")
        checkpoint_path = f"{config['save_dir']}/{config['old-problem']}_pretrained_{config['Model_Type']}_best.pth"
        checkpoint = torch.load(checkpoint_path)
        
        logger.info(f"Loading the checkpoint from {checkpoint_path}")
        ts2vec.load_state_dict(checkpoint['state_dict'])

        # update the save_dir to save the resumed training
        path_levels = config['save_dir'].split('/')
        path_levels[-2] = datetime.now().strftime("%Y-%m-%d_%H-%M")
        config['save_dir'] = '/'.join(path_levels)
        logger.info(f"Saving the resumed training in {config['save_dir']}")


    since = datetime.now()
    _, logs = ts2vec.fit_ssl(train_loader=train_loader, 
                                         val_loader=test_loader,
                                         optimizer=ts2vec_optim, 
                                         config=config)
    ssl_time = datetime.now() - since
    logs['time'] = ssl_time.total_seconds()

    logger.info(f"Best validation loss: {logs['val_losses'][logs['best_epoch']-1]}")
    logger.info(f"Saved best model, the model from epoch {logs['best_epoch']}\n")
    logger.info(f"\nFinished! Training time {ssl_time}")
    
    return config, logs

from datetime import datetime
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .tc import TC
from .model import BaseModel as Encoder
from .dataloader import Load_Dataset
from .augmentations import ConfigAug


logger = logging.getLogger('__main__')


def pre_training(config, data, resume_train = False):
    '''Pretrain the TS-TCC feature extractor model in the pretext task.

    Args:
        config (dict): Trainig configuration.
        data (dict): Train dataset.
        resume_train (bool): Resume training from a checkpoint. Default = False.

    Returns:
        pretrained TS-TCC params (dict), training logs (dict)
    '''
    model_args = config['model_args']

    # create agumented versions of the datasets -------------------------------------------
    logger.info("Augmenting the dataset ...")
    config_aug = ConfigAug(**model_args['augmentation'])
    
    train_dataset = Load_Dataset(X_train=data['train_data'], 
                                 y_train=data['train_label'],
                                 config=config_aug, 
                                 training_mode="self_supervised")

    val_dataset = Load_Dataset(X_train=data['test_data'],
                               y_train=data['test_label'], 
                               config=config_aug, 
                               training_mode="self_supervised")

    logger.info(f"train shape {train_dataset.x_data.shape} / test shape {val_dataset.x_data.shape}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, drop_last=True)

    # instanciate the model ---------------------------------------------------------------
    logger.info("Creating the TS2Vec Self-supervised model ...")
    tc = TC(device=config['gpu'], **model_args['tc_model']) # temporal contrasting module
    tc_optim = optim.Adam(tc.parameters(), **config['optim_args'])

    encoder = Encoder(**model_args['encoder']) # feature extractor module
    encoder_optim = torch.optim.Adam(encoder.parameters(), **config['optim_args'])

    # load the checkpoint if resume training ---------------------------------------------
    if resume_train:
        logger.info("Resuming the training ...")
        checkpoint_path = f"{config['save_dir']}/{config['old-problem']}_pretrained_{config['Model_Type']}_last.pth"
        checkpoint = torch.load(checkpoint_path)
        
        logger.info(f"Loading the checkpoint from {checkpoint_path}")
        encoder.load_state_dict(checkpoint['encoder'])
        tc.load_state_dict(checkpoint['tc_model'])

        # update the save_dir to save the resumed training
        path_levels = config['save_dir'].split('/')
        path_levels[-2] = datetime.now().strftime("%Y-%m-%d_%H-%M")
        config['save_dir'] = '/'.join(path_levels)
        logger.info(f"Saving the resumed training in {config['save_dir']}")


    # start training ----------------------------------------------------------------------
    since = datetime.now()
    logger.info(f"Self-supervised TS-TCC pretraining ...")

    _, logs = tc.fit_ssl(train_loader,
                         test_loader,
                         tc_optim,
                         encoder,
                         encoder_optim,
                         config,
                         **model_args['cc_loss'])
    time = datetime.now() - since
    logs['time'] = time.total_seconds()

    logger.info(f"Best validation loss: {logs['val_losses'][logs['best_epoch']-1]}")
    logger.info(f"Saved best model, the model from epoch {logs['best_epoch']}")
    logger.info(f"Finished! Training time {time}")

    return config, logs

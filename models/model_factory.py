import logging
from models.Series2Vec import Series2Vec
from models.TSTCC import Encoder as TSTCC_encoder
from models.TS2Vec import TS2Vec


logger = logging.getLogger('__main__')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Model_factory(config, data):
    config['Data_shape'] = data['train_data'].shape
    config['num_labels'] = int(max(data['train_label'])) + 1

    if config['Model_Type'] == 'Series2Vec':
        model = Series2Vec.Seires2Vec(config, num_classes=config['num_labels'])
    
    if config['Model_Type'] == 'TSTCC':
        model = TSTCC_encoder(**config['model_args']['encoder'])

    if config['Model_Type'] == 'TS2Vec':
        model = TS2Vec(**config['model_args'])    

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    return model

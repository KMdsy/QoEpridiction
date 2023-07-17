# data configs
train_cfg = {
    'path': './data',
    'dataset': 'cellular',
    'n_features': 10 * 24, # 10 days
    'n_pred': 5 * 24, # 5 days
    'batch_size': 1
}
    
data_cfg = {
    'data_file': 'data/cellular/cellular.csv',
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'transfer_entropy_delay': 1
}
    
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

train_configs = Config(**train_cfg)
data_configs = Config(**data_cfg)
patchtst_config = {
    'input_size': 52,           
    'h': 53,                  
    'max_steps': 60 * 104,    
    'batch_size': 64,
    'random_seed': 42,
    'activation': 'relu',     
    'dropout': 0.2,           
}


nbeats_config = {
    'max_steps': 25 * 104,
    'h': 53,
    'random_seed': 42,
    'input_size':52,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'shared_weights':True,
    'optimizer': 'torch.optim.AdamW',
    'activation': 'ReLU'

}

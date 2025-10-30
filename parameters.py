
from dataclasses import dataclass, field
import torch

@dataclass
class Params:

    # transformer parameters
    dim_embed=128
    dim_ff = 512 
    num_heads=4 
    num_features =  30
    seq_len = 40
    num_layers = 2
    num_classes = 2
    dropout=0.1
    
    # training parameters
    batch_size : int = 10
    criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.CrossEntropyLoss() for multi-classes 
    
    shuffle = True
    learning_rate = 5e-4
    n_epochs = 5
    train_steps = 1
    val_steps = 1
    checkpoint_frequency = 1000

    model_name = 'SkipGram'
    model_dir = "weights/{}".format(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
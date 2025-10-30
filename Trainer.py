from tqdm import tqdm
from time import monotonic 
import random
import os
import json

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, BinaryAccuracy
from typing import Union

from TransformerTimeseriesClassifier import *
from CNNTimeseriesClassifier import *
# from LSTMTimeseriesClassifier import **

class Trainer:
    def __init__(self, model: Union[SelfAttentionEncoderClassifier,ResNetClassifier], params: Params, 
                 optimizer, train_iter, valid_iter, debug=False):
        self.model = model
        self.debug = debug
        self.params = params
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        self.epoch_train_mins = {}
        self.loss = {"train": [], "valid": []}
        self.accuracy = {"train": [], "valid": []}
        
        # sending all to device
        self.model.to(self.params.device)
        self.test_tokens = None
        self.model_path = 'model.pth'
        self.loss_path = 'loss.json'
   
    
    def train(self):
        self.do_test()
        for epoch in range(self.params.n_epochs):
            # load data
            self.train_dataloader = DataLoader(
                self.train_iter,
                batch_size=self.params.batch_size,
                shuffle=True,
            )
            self.valid_dataloader = DataLoader(
                self.valid_iter,
                batch_size=self.params.batch_size,
                shuffle=False,
            )
            
            # train model
            st_time = monotonic()
            self._train_epoch(epoch)
            self.epoch_train_mins[epoch] = round((monotonic()-st_time)/60, 1)

            # validate model
            self._validate_epoch(epoch)
            print(f"""Epoch: {epoch+1}/{self.params.n_epochs}\n""",
            f"""    Train Loss: {self.loss['train'][-1]:.2}, Accuracy: {self.accuracy['train'][-1]:.2}\n""",
            f"""    Valid Loss: {self.loss['valid'][-1]:.2}, Accuracy: {self.accuracy['valid'][-1]:.2}\n""",
            f"""    Training Time (mins): {self.epoch_train_mins.get(epoch)}"""
            """\n"""
            )

            if self.params.checkpoint_frequency:
                self._save_checkpoint(epoch)

        self.save_loss()
        self.do_test(save_results=True)  

    
    def _train_epoch(self,_epoch):
        self.model.train()
        running_loss = []
        accuracy = BinaryAccuracy().to(self.params.device)        
        #accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device) # for multi-classes
        
        progress = tqdm(self.train_dataloader, desc=f"Epoch {_epoch+1}/{self.params.n_epochs}")
        for i, batch_data in enumerate(progress, 1):                    
            X_batch, y_batch = batch_data[0], batch_data[1]   

            X_batch = X_batch.to(self.params.device)
            y_batch = y_batch.to(self.params.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch).squeeze()                # Forward
            loss = self.params.criterion(logits, y_batch.float()) # Compute loss
            #loss = self.params.criterion(logits, y_batch)        # for multi-classes with nn.CrossEntropyLoss()
           
            if self.debug >1 :
                before = self.model.encoder.layers[0].self_attn.W_Q.weight.clone()

            loss.backward()
            self.optimizer.step()
            
            if self.debug >1 :
                after = self.model.encoder.layers[0].self_attn.W_Q.weight
                print("Change:", (after - before).abs().mean())
                for name, param in self.model.named_parameters():
                    if "self_attn.W_Q" in name:
                        print(name, param.grad is None, param.grad.abs().mean().item() if param.grad is not None else None)

            # Save and print loss
            running_loss.append(loss.item())
            progress.set_postfix(loss=loss.item())

            # Compute accuracy
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                acc = accuracy(probs, y_batch) 
                #preds = (probs > 0.5).long()       # for multi-classes with nn.CrossEntropyLoss()
                #acc = accuracy(preds, y_batch)           
        
        epoch_loss = np.mean(running_loss)
        epoch_acc = accuracy.compute()  # aggregate over epoch
        accuracy.reset()  # reset for next epoch
        
        self.loss['train'].append(epoch_loss)
        self.accuracy['train'].append(epoch_acc) 


    def _validate_epoch(self,_epoch):
        self.model.eval()
        running_loss = []
        accuracy = BinaryAccuracy().to(self.params.device)    

        with torch.no_grad():
            progress = tqdm(self.valid_dataloader, desc=f"Epoch {_epoch+1}/{self.params.n_epochs}")
            for i, batch_data in enumerate(progress, 1): 
                X_batch, y_batch = batch_data[0], batch_data[1]   

                X_batch = X_batch.to(self.params.device)
                y_batch = y_batch.to(self.params.device)
            
                logits = self.model(X_batch).squeeze()           # Forward
                loss = self.params.criterion(logits, y_batch)    # Compute loss
                
                running_loss.append(loss.item())
                progress.set_postfix(loss=loss.item())

                # Compute accuracy
                probs = torch.sigmoid(logits)
                acc = accuracy(probs, y_batch) 
                
            epoch_loss = np.mean(running_loss)
            epoch_acc = accuracy.compute()  # aggregate over epoch
            accuracy.reset()  # reset for next epoch   
            
            self.loss['valid'].append(epoch_loss)
            self.accuracy['valid'].append(epoch_acc) 


    def compute_feature_importance(self, x, target_class=None):
        """
        x: tensor of shape (1, seq_len, num_features)
        target_class: int, class index to compute gradients w.r.t
        """
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self.model(x)  # shape: (batch_size,)

        
        # Backward pass
        self.model.zero_grad() 
        
        # Option 1: use the raw logit (common)
        #logit = logits[0]
        #logit.backward()

        # Option 2: use sigmoid(logit) if you prefer probability-based gradients
        torch.sigmoid(logits[0]).backward()

        # Gradients w.r.t input
        saliency = x.grad.abs()
        saliency = saliency / (saliency.max() + 1e-8)
        saliency = saliency.mean(dim=1) # average over sequence → shape: (num_features,)
        return saliency.detach()


    def do_test(self, save_results=False, print_test=True):
        assert next(self.model.parameters()).device == self.params.device, "Device mis-match"
        return
        
    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.params.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.params.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_path`"""
        torch.save(self.model.state_dict(), self.model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.loss_path`"""   
        with open(self.loss_path, "w") as fp:
            json.dump(self.loss, fp)


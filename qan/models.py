from easydict import EasyDict as edict
import os.path as osp

from qan.models.seq2seq import Seq2Seq, Encoder, Decoder
from qan.utils.initializer import initialize_optimizer
from qan.utils.misc import load_pretrained_embedding

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch import nn

        
        
class Seq2SeqQA(pl.LightningModule):
    def __init__(self, 
                 model_cfg: edict,        # model configuration
                 optimizer_cfg: edict,    # opitimizer configuration
                ):
        super().__init__()

        # set configs
        self.save_hyperparameters()
        self.optimizer = initialize_optimizer(torch.optim, optimizer_cfg.module)
        self.criterion = nn.CrossEntropyLoss()
        
        # ------------
        # Model
        # ------------
        self.model_cfg = model_cfg
        self.encoder = Encoder(**self.model_cfg.encoder)
        self.decoder = Decoder(**self.model_cfg.decoder)
        self.classifier = nn.Linear(in_features=self.decoder.hidden_dim,
                                out_features=self.decoder.num_embeddings)
        
        assert self.encoder.hidden_dim * 2 == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"
            
                  
        # pretrained
        if osp.isfile(model_cfg.args.pretrained):
            print("load pretrained embedding")
            pretrained_embedding = load_pretrained_embedding(model_cfg.args.pretrained, 
                                                             vocab, num_embedding=model_cfg.args.num_embedding)
            self.model.load_embedding(pretrained_embedding)
        else:
            print("Train without pretrained embedding")
        
        
    
    def predict(self, x):
        out = self.model(x)
        return self.postprocess(out)
    
    def postprocess(self, x):
        pass
    
    def forward(self, context: torch.Tensor, question: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5):
        """[pass forward method of the model]

        Args:
            context (torch.Tensor): [context tensor, shape: batch_size, padded_variable_len]
            question (torch.Tensor): [question tensor, shape: batch_size, padded_variable_len]
            teacher_forcing_ratio (float, optional): [threshold to use ground truth as decoder inputs]. 
                                                      Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        
        batch_size, context_seq_len = context.shape
        question_seq_len = question.shape[1]
        
        # set to 2 due to predict the start,end over possibles location in the context
        decoding_sequence = 2 
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # encode context and question
        _, context_hidden, context_cell = self.encoder(context)
        _, question_hidden, question_cell = self.encoder(question)
        
        # concate the context and question hidden states
        # context_hidden: num_layers * num_directions(2 for bidirectional), batch_size, hidden_dim
        # question_hidden: num_layers * num_directions(2 for bidirectional), batch_size, hidden_dim
        hidden_state_cat = torch.cat(context_hidden, question_hidden, dim=2)
        cell_state_cat = torch.cat(context_cell, question_cell)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(decoding_sequence):
            
            output_decoder, hidden_state_cat, cell_state_cat = self.decoder(input, hidden_state_cat, cell_state_cat)
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
    
    
    def training_step(self, batch: set, batch_idx):
        """[training step]

        Args:
            batch ([set]): [input, target, length input]
            batch_idx ([type]): [description]

        Returns:
            loss[float]: [loss value]
        """        
        x, y, len_x = batch
        y = y.to(torch.long)

        # inferencing
        logits = self(x, len_x)
        preds = F.softmax(logits, dim=1)

        loss = self.criterion(logits, y)

        # log loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # calculate metrics & log metrics
        for metric in self.metrics.train:
            y = y.to(torch.int)
            self.log(f'train_{metric}', self.metrics.train[metric](preds.cpu(), y.cpu()), 
                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, len_x = batch
        y = y.to(torch.long)

        # inferencing
        logits = self.forward(x, len_x)
        preds = F.softmax(logits, dim=1)

        # calculate loss
        loss = self.loss(logits, y)

        # log 
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        # calculate metrics & log metrics
        for metric in self.metrics.val:
            self.log(f'val_{metric}', self.metrics.val[metric](preds.cpu(), y.cpu()), 
                    on_epoch=True, prog_bar=True, logger=True, on_step=True)
 

    def test_step(self, batch, batch_idx):
        # get vars
        x, y, len_x = batch
        y = y.to(torch.long)

        # inferencing
        logits = self.forward(x, len_x)
        preds = F.softmax(logits, dim=1)

        # calculate loss
        loss = self.loss(logits, y)

        # log 
        self.log("test_loss", loss, on_step=True, on_epoch=True)

        # calculate metrics & log metrics
        for metric in self.metrics.test:
            self.log(f'test_{metric}', self.metrics.test[metric](preds.cpu(), y.cpu()), 
                    on_epoch=True, prog_bar=True, logger=True, on_step=True)
                
    def validation_epoch_end(self, *args, **kwargs):
        # self.log_dict(self.metrics.compute(), on_epoch=True, prog_bar=True, logger=True)
        # # we know that ClassificationMetrics' state need to be reset
        # self.metrics.eval_init()
        pass

    def training_epoch_end(self, outs):
        # # log epoch metric
        # self.log_dict(self.metrics.compute())
        # # we know that ClassificationMetrics' state need to be reset
        # self.metrics.eval_init()
        pass
                   
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.hparams.optimizer_cfg)
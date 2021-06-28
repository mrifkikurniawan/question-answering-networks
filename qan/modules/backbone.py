import random
from easydict import EasyDict as edict

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 num_layers: int,
                 pretrained: edict=False,
                 embedding_dim: int=300, 
                 hidden_dim: int=512,  
                 dropout:float=0.5,
                 bidirectional: bool=True, 
                 embedding_kwargs: dict={},
                 rnn_kwargs: dict={}
                 ): 
        """[LSTM-based Encoder]

        Args:
            num_embeddings (int): [embedding number]
            num_layers (int): [number of RNN layers]
            embedding_dim (int, optional): [embedding dimensions]. Defaults to 300.
            hidden_dim (int, optional): [RNN hidden dimensions]. Defaults to 512.
            dropout (float, optional): [dropout probability]. Defaults to 0.5.
            bidirectional (bool, optional): [flag for bidirectional LSTM]. Defaults to True.
            embedding_kwargs (dict, optional): [additional arguments for embbeding layer]. Defaults to {}.
            rnn_kwargs (dict, optional): [additional arguments for rnn layer]. Defaults to {}.
        """    
            
        super().__init__()  
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_embeddings = num_embeddings
        self.bidirectional = bidirectional
        
        if pretrained:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim,
                                        **embedding_kwargs).from_pretrained(pretrained.tensor, freeze=pretrained.freeze)
        else:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim,
                                        **embedding_kwargs) 
        
        self.rnn = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_dim, 
                           num_layers=num_layers, 
                           dropout=dropout, 
                           batch_first=True, 
                           bidirectional=bidirectional, 
                           **rnn_kwargs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor):
        """[forward function of the encoder]

        Args:
            input ([torch.Tensor]): [input tensor (shape: batch_size, variable_len]
        Returns:
            outputs [torch.Tensor]: [outputs of RNN (shape: batch_size, sequence_length, num_directions(2 for bidirectional) * hidden_dim)]
            hidden [torch.Tensor]: [hidden state of RNN (shape: num_layers * num_directions(2 for bidirectional), batch_size, hidden_dim)]
            cell [torch.Tensor]: [cell state of RNN (shape: num_layers * num_directions(2 for bidirectional), batch_size, hidden_dim)]
        """        
        
        embedding = self.dropout(self.embedding(input)) # batch_size, sequence_length, dim

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs: batch_size, sequence_length, num_directions(2 for bidirectional) * hidden_dim
        # hidden: num_layers * num_directions(2 for bidirectional), batch size, hidden_dim
        # cell: num_layers * num_directions(2 for bidirectional), batch size, hidden_dim
        
         
        return outputs, hidden, cell
    
    @property
    def hidden_dim(self):
        return self._hidden_dim
    
    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def num_layers(self):
        return self._num_layers
    
    

class Decoder(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 num_layers: int,
                 embedding_dim: int=300, 
                 hidden_dim: int=512,  
                 dropout:float=0.5,
                 bidirectional: bool=True,
                 embedding_kwargs: dict={},
                 rnn_kwargs: dict={}
                 ):
        """[summary]

        Args:
            num_embeddings (int): [embedding number]
            num_layers (int): [number of RNN layers]
            embedding_dim (int, optional): [embedding dimensions]. Defaults to 300.
            hidden_dim (int, optional): [RNN hidden dimensions]. Defaults to 512.
            dropout (float, optional): [dropout probability]. Defaults to 0.5.
            bidirectional (bool, optional): [flag for bidirectional LSTM]. Defaults to True.
            embedding_kwargs (dict, optional): [additional arguments for embbeding layer]. Defaults to {}.
            rnn_kwargs (dict, optional): [additional arguments for rnn layer]. Defaults to {}.
        """        
        
        super().__init__()
        self._num_embeddings = num_embeddings
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, **embedding_kwargs)
        self.rnn = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_dim, 
                           num_layers=num_layers, 
                           dropout=dropout, 
                           batch_first=True, 
                           bidirectional=bidirectional, 
                           **rnn_kwargs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        """[forward funtion of the decoder]

        Args:
            input ([torch.Tensor]): [input tensor (shape: batch_size, variable_len or == 1 for autoregressive decoder]
        Returns:
            outputs [torch.Tensor]: [outputs of RNN (shape: batch_size, sequence_length, num_directions(2 for bidirectional) * hidden_dim)]
            hidden [torch.Tensor]: [hidden state of RNN (shape: num_layers * num_directions(2 for bidirectional), batch_size, hidden_dim)]
            cell [torch.Tensor]: [cell state of RNN (shape: num_layers * num_directions(2 for bidirectional), batch_size, hidden_dim)]
        """   
        
        embedding = self.dropout(self.embedding(input)) # batch_size, sequence_length/1, dim    
               
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) 
        # outputs: batch_size, sequence_length, num_directions(2 for bidirectional) * hidden_dim
        # hidden: num_layers * num_directions(2 for bidirectional), batch size, hidden_dim
        # cell: num_layers * num_directions(2 for bidirectional), batch size, hidden_dim
        
        return outputs, hidden, cell

    @property
    def hidden_dim(self):
        return self._hidden_dim
    
    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def num_layers(self):
        return self._num_layers
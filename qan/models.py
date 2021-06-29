from easydict import EasyDict as edict
from typing import List

from qan.modules import Encoder, Decoder
from qan.utils.initializer import initialize_optimizer, create_instance, initialize_pretrained
from qan.modules import SimpleMLP 

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.init import xavier_uniform
from transformers.tokenization_utils_base import BatchEncoding
import transformers

    
# -----------------------------------
# Seq2Seq RNN
# -----------------------------------
class Seq2SeqQA(pl.LightningModule):
    def __init__(self, 
                 model_cfg: edict,        # model configuration
                 optimizer_cfg: edict,    # opitimizer configuration
                 tokenizer_cfg: edict,    # tokenizer configuration
                 pretrained_cfg: edict=None
                ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.pretrained_cfg = pretrained_cfg
        
        # ------------
        # Training Building Block
        # ------------    
        self.tokenizer = initialize_pretrained(tokenizer_cfg)
        self.vocab = edict(self.tokenizer.get_vocab())
        self.vocab_size = self.tokenizer.vocab_size 
        self.max_sequence = self.tokenizer.model_max_length
        self.optimizer = initialize_optimizer(torch.optim, optimizer_cfg.module)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.max_sequence)
                  
        # ------------
        # Pre-trained
        # ------------
        if pretrained_cfg:
            print("load pretrained embedding")
            pretrained_embedding = self._load_pretrained_embedding(pretrained_cfg=pretrained_cfg, 
                                                                   embedding_dim=model_cfg.encoder.embedding_dim,
                                                                   vocab=self.vocab)
            pretrained_cfg.tensor = pretrained_embedding
        else:
            print("Train without pretrained embedding")
            

        self._init_model()
    
    
    def _init_model(self):
        # Var
        bidirectional = self.model_cfg.encoder.bidirectional
        encoder_hidden = self.model_cfg.encoder.hidden_dim
        dir = 2 if bidirectional else 1
        
        # ------------
        # Backbone
        # ------------
        self.encoder = Encoder(num_embeddings=self.vocab_size, 
                               pretrained=self.pretrained_cfg, 
                               **self.model_cfg.encoder)
        
        # ------------
        # Classifier
        # ------------
        classifier_in_features = encoder_hidden * dir 
        self.classifier = SimpleMLP(in_features=classifier_in_features,
                                          out_features=2)
        
        
    def _preprocess_train_batch(self, batch):
        
        batch = edict(batch)
        answers = batch.answers
        encodings = self.tokenizer(batch.context, 
                                   batch.question,
                                   truncation=True,
                                   padding=True)
        
        # get answers start/end positions in encoding
        answers_start_positions, answers_end_positions = self._get_answers_token_positions(context_encodings=encodings, 
                                                                                           answers=answers)
                 
        return encodings, answers_start_positions, answers_end_positions    

    
    def _preprocess_inference(self, batch):
        batch = edict(batch)
        encodings = self.tokenizer(batch.context, 
                                   batch.question,
                                   truncation=True,
                                   padding=True)
        encodings_ids = torch.tensor(encodings['input_ids'], dtype=torch.int32, device=self.device)
        
        return encodings_ids, batch.context, encodings
    
    @torch.no_grad()
    def predict(self, x):
        encodings_ids, gold_context, encodings = self._preprocess_inference(x)
        out = self(inputs=encodings_ids)
        
        logits_start, logits_end = out
        predictions_start, predictions_end = F.softmax(logits_start, dim=1), F.softmax(logits_end, dim=1) # (batch_size, num_classes/max_sequence) 
        
        return self.postprocess((predictions_start, predictions_end), gold_context=gold_context, encodings=encodings)
    
    def postprocess(self, 
                    start_end_pair: set, 
                    gold_context: list,
                    encodings: BatchEncoding) -> List:
        # get confidence and labels
        predictions_start, predictions_end = start_end_pair
        confs_start, preds_start = torch.max(predictions_start, dim=1)
        confs_end, preds_end = torch.max(predictions_end, dim=1)
        
        # convert to numpy tensor
        confs_start, preds_start = confs_start.cpu().numpy(), preds_start.cpu().numpy()
        confs_end, preds_end = confs_end.cpu().numpy(), preds_end.cpu().numpy() 
        
        outputs = list()    
        for i in range(len(preds_start)):
            start = preds_start[i]
            end = preds_end[i]  
            
            start_str_span = encodings.token_to_chars(i, start) # CharSpan
            end_str_span = encodings.token_to_chars(i, end)     # CharSpan
            
            ans = gold_context[i][start_str_span.start:end_str_span.end]
            outputs.append(ans)

        return outputs
    
    def forward(self, 
                inputs: torch.Tensor):
        """[pass forward method of the model]

        Args:
            inputs (torch.Tensor): [concatination of context and question ids, shape: batch_size, padded_variable_len]


        Returns:
            [type]: [description]
        """
        
        # encode context and question
        out_context, _, _ = self.encoder(inputs) # n_layers * n_directions(2 for bidirectional), bs, hid_dim
                
        logits = self.classifier(out_context) # batch_size, max_sequence, num_classes==2
        start_logits, end_logits = logits.split(1, dim=-1) # batch_size, max_sequence, 1
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        
        return start_logits, end_logits
    
    def training_step(self, batch: set, batch_idx):
        """[training step]

        Args:
            batch ([set]): [input, target, length input]
            batch_idx ([type]): [description]

        Returns:
            loss[float]: [loss value]
        """        
        encodings, answers_start_positions, answers_end_positions  = self._preprocess_train_batch(batch)
        # questions_ids, _ = questions_encodings['input_ids'], questions_encodings['attention_mask']
        # contexts_ids, _ = contexts_encodings['input_ids'], contexts_encodings['attention_mask']
        encodings_ids = encodings['input_ids']
        
        # pass forward the model
        start_logits, end_logits = self(torch.tensor(encodings_ids, dtype=torch.int32, device=self.device))
        
        # calculate loss value
        start_logits = start_logits.squeeze(-1).contiguous()  # (batch_size, class)
        end_logits = end_logits.squeeze(-1).contiguous()  # (batch_size, class)
        
        # move targets to device
        answers_start_positions = answers_start_positions.to(self.device)
        answers_end_positions = answers_end_positions.to(self.device)
        
        # measure loss
        start_loss = self.criterion(start_logits, answers_start_positions)
        end_loss = self.criterion(end_logits, answers_end_positions)
        loss = (start_loss + end_loss) / 2

        # log loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def _get_answers_token_positions(self, 
                             context_encodings, 
                             answers: edict):
        start_positions = []
        end_positions = []
        
        answer_start = answers.answer_start[0]
        answer_end = answers.answer_end[0]
        answer_text = answers.text[0]
        
        assert isinstance(answer_start, torch.Tensor)
        assert isinstance(answer_end, torch.Tensor)
        
        for i in range(len(answer_text)):
            start = context_encodings.char_to_token(i, answer_start[i])
            end = context_encodings.char_to_token(i, answer_end[i] - 1)

            start_positions.append(start)
            end_positions.append(end)            

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
                start = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
                end = self.tokenizer.model_max_length
        
        return torch.tensor(start_positions), torch.tensor(end_positions) 

    def _load_pretrained_embedding(self, 
                                   pretrained_cfg: edict, 
                                   embedding_dim: int,
                                   vocab: edict
                                   ):
        vecs = create_instance(pretrained_cfg)
        pretrained_tensor = torch.zeros((len(vocab), embedding_dim))
        
        for i, word in enumerate(vocab):
            id = vocab[word]
            vec = vecs.get_vecs_by_tokens(word, lower_case_backup=True)
            
            # if unknwon, initialize embedding
            if bool(torch.sum(vec) == 0.):
                pretrained_tensor[id, :] = xavier_uniform(vec.unsqueeze(0))
            else:
                pretrained_tensor[id, :] = vec.unsqueeze(0)
                
        return pretrained_tensor

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.hparams.optimizer_cfg.args)


    

# -----------------------------------
# Transformers-based QA Model
# -----------------------------------
class TransformersQA(pl.LightningModule):
    def __init__(self, 
                 model_cfg: edict,        # model configuration
                 optimizer_cfg: edict,    # opitimizer configuration
                 tokenizer_cfg: edict,    # tokenizer configuration
                ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        
        # ------------
        # Training Building Block
        # ------------    
        self.tokenizer = initialize_pretrained(tokenizer_cfg)
        self.vocab = edict(self.tokenizer.get_vocab())
        self.vocab_size = self.tokenizer.vocab_size 
        self.max_sequence = self.tokenizer.model_max_length
        self.optimizer = initialize_optimizer(transformers, optimizer_cfg.module)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.max_sequence)
        
        # init model
        self._init_model()
    
    def _init_model(self):
        # ------------
        # Backbone
        # ------------
        encoder_config = self.model_cfg.encoder.config
        self.tranformer_config = create_instance(encoder_config)
        self.encoder = initialize_pretrained(self.model_cfg.encoder)
        
        # ------------
        # Classifier
        # ------------
        self.classifier = SimpleMLP(in_features=encoder_config.args.hidden_size,
                                    out_features=2)
        
        
    def _preprocess_train_batch(self, batch):
        
        batch = edict(batch)
        answers = batch.answers
        encodings = self.tokenizer(batch.context, 
                                   batch.question,
                                   truncation=True,
                                   padding=True)
        
        # get answers start/end positions in encoding
        answers_start_positions, answers_end_positions = self._get_answers_token_positions(context_encodings=encodings, 
                                                                                           answers=answers)
                 
        return encodings, answers_start_positions, answers_end_positions    

    
    def _preprocess_inference(self, batch):
        batch = edict(batch)
        encodings = self.tokenizer(batch.context, 
                                   batch.question,
                                   truncation=True,
                                   padding=True)
        encodings_ids = torch.tensor(encodings['input_ids'], dtype=torch.int32, device=self.device)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.int32, device=self.device)
        
        return (encodings_ids, attention_mask), batch.context, encodings
    
    @torch.no_grad()
    def predict(self, x):
        encodings_ids, attention_mask, gold_context, encodings = self._preprocess_inference(x)
        
        # pass forward the model
        out = self(input_ids = encodings_ids,
                   attention_mask = attention_mask)
        
        logits_start, logits_end = out
        predictions_start, predictions_end = F.softmax(logits_start, dim=1), F.softmax(logits_end, dim=1) # (batch_size, num_classes/max_sequence) 
        
        return self.postprocess((predictions_start, predictions_end), gold_context=gold_context, encodings=encodings)
    
    def postprocess(self, 
                    start_end_pair: set, 
                    gold_context: list,
                    encodings: BatchEncoding) -> List:
        # get confidence and labels
        predictions_start, predictions_end = start_end_pair
        confs_start, preds_start = torch.max(predictions_start, dim=1)
        confs_end, preds_end = torch.max(predictions_end, dim=1)
        
        # convert to numpy tensor
        confs_start, preds_start = confs_start.cpu().numpy(), preds_start.cpu().numpy()
        confs_end, preds_end = confs_end.cpu().numpy(), preds_end.cpu().numpy() 
        
        outputs = list()    
        for i in range(len(preds_start)):
            start = preds_start[i]
            end = preds_end[i]  
            
            start_str_span = encodings.token_to_chars(i, start) # CharSpan
            end_str_span = encodings.token_to_chars(i, end)     # CharSpan
            
            ans = gold_context[i][start_str_span.start:end_str_span.end]
            outputs.append(ans)

        return outputs
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """[pass forward method of the model]

        Args:
            input_ids (torch.Tensor): [concatination of context and question ids, shape: (batch_size, max_sequence)]
            attention_mask (torch.Tensor): []
        Returns:
            start_logits[torch.Tensor]: [logits of start indexes predictions, shape: (batch_size, max_sequence)]
            end_logits[torch.Tensor]: [logits of end indexes predictions, shape: (batch_size, max_sequence)]
        """
                
        # encode context and question
        outputs = self.encoder(input_ids = input_ids,
                               attention_mask = attention_mask) 
        last_hidden_state = outputs[0] # batch_size, max_sequence, hidden_dim
        
        logits = self.classifier(last_hidden_state) # batch_size, max_sequence, num_classes==2
        start_logits, end_logits = logits.split(1, dim=-1) # batch_size, max_sequence, 1
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        
        return start_logits, end_logits
    
    def training_step(self, batch: set, batch_idx):
        """[training step]

        Args:
            batch ([set]): [input, target, length input]
            batch_idx ([type]): [description]

        Returns:
            loss[float]: [loss value]
        """        
        encodings, answers_start_positions, answers_end_positions  = self._preprocess_train_batch(batch)
        encodings_ids = torch.tensor(encodings['input_ids'], dtype=torch.int32, device=self.device)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.int32, device=self.device)
        
        # pass forward the model
        start_logits, end_logits = self(input_ids = encodings_ids,
                                        attention_mask = attention_mask)
        
        # calculate loss value
        start_logits = start_logits.squeeze(-1).contiguous()  # (batch_size, class)
        end_logits = end_logits.squeeze(-1).contiguous()  # (batch_size, class)
        
        # move targets to device
        answers_start_positions = answers_start_positions.to(self.device)
        answers_end_positions = answers_end_positions.to(self.device)
        
        # measure loss
        start_loss = self.criterion(start_logits, answers_start_positions)
        end_loss = self.criterion(end_logits, answers_end_positions)
        loss = (start_loss + end_loss) / 2

        # log loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def _get_answers_token_positions(self, 
                             context_encodings, 
                             answers: edict):
        start_positions = []
        end_positions = []
        
        answer_start = answers.answer_start[0]
        answer_end = answers.answer_end[0]
        answer_text = answers.text[0]
        
        assert isinstance(answer_start, torch.Tensor)
        assert isinstance(answer_end, torch.Tensor)
        
        for i in range(len(answer_text)):
            start = context_encodings.char_to_token(i, answer_start[i])
            end = context_encodings.char_to_token(i, answer_end[i] - 1)

            start_positions.append(start)
            end_positions.append(end)            

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
                start = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
                end = self.tokenizer.model_max_length
        
        return torch.tensor(start_positions), torch.tensor(end_positions) 

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.hparams.optimizer_cfg.args)
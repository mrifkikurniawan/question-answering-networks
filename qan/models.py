from easydict import EasyDict as edict
from typing import List

from qan.modules import Encoder, Decoder
from qan.utils.initializer import initialize_optimizer, create_instance, initialize_tokenizer
from qan.modules import SimpleMLP 

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.init import xavier_uniform

        
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
        self.tokenizer = initialize_tokenizer(tokenizer_cfg)
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
        encoder_num_layers = self.model_cfg.encoder.num_layers
        dir = 2 if bidirectional else 1
        
        # ------------
        # Backbone
        # ------------
        self.question_encoder = Encoder(num_embeddings=self.vocab_size, 
                               pretrained=self.pretrained_cfg, 
                               **self.model_cfg.encoder)
        self.context_encoder = Encoder(num_embeddings=self.vocab_size, 
                               pretrained=self.pretrained_cfg, 
                               **self.model_cfg.encoder)
        
        self.end2end_encoder = nn.LSTM(input_size=dir*encoder_hidden, 
                                       hidden_size=encoder_hidden, 
                                       num_layers=encoder_num_layers, 
                                       dropout=0.25, 
                                       batch_first=True, 
                                       bidirectional=True, 
                                       )
        
        # ------------
        # Classifier
        # ------------
        classifier_in_features = encoder_hidden * dir 
        self.classifier = SimpleMLP(in_features=classifier_in_features,
                                          out_features=2)
        
        
    def _preprocess_train_batch(self, batch):
        
        batch = edict(batch)
        answers = batch.answers
        questions_train_encodings = self.tokenizer(batch.question, 
                                                   truncation=True, 
                                                   padding=True)
        contexts_train_encodings = self.tokenizer(batch.context, 
                                                  truncation=True, 
                                                  padding=True)
        
        # get answers start/end positions in encoding
        answers_start_positions, answers_end_positions = self._get_answers_token_positions(context_encodings=contexts_train_encodings, 
                                                                                           answers=answers)
                 
        return questions_train_encodings, contexts_train_encodings, answers_start_positions, answers_end_positions    


    def _preprocess_val_batch(self, batch):
        return self._preprocess_train_batch(batch)
    
    def _preprocess_inference(self, batch):
        batch = edict(batch)
        questions_encodings = self.tokenizer(batch.question, 
                                             truncation=True, 
                                             padding=True)
        contexts_encodings = self.tokenizer(batch.context, 
                                            truncation=True, 
                                            padding=True)

        questions_ids, _ = questions_encodings['input_ids'], questions_encodings['attention_mask']
        contexts_ids, _ = contexts_encodings['input_ids'], contexts_encodings['attention_mask']
        
        questions_ids = torch.tensor(questions_ids, dtype=torch.int32, device=self.device) # (batch_size, max_sequence_per_batch)
        contexts_ids = torch.tensor(contexts_ids, dtype=torch.int32, device=self.device) # (batch_size, max_sequence_per_batch)
        
        return questions_ids, contexts_ids
    
    @torch.no_grad()
    def predict(self, x):
        questions_ids, contexts_ids = self._preprocess_inference(x)
        out = self(context=contexts_ids,
                   question=questions_ids)
        
        logits_start, logits_end = out
        predictions_start, predictions_end = F.softmax(logits_start, dim=1), F.softmax(logits_end, dim=1) # (batch_size, num_classes/max_sequence) 
        
        return self.postprocess((predictions_start, predictions_end), contexts_ids=contexts_ids)
    
    def postprocess(self, 
                    x: set, 
                    contexts_ids: List[List]) -> List:
        # get confidence and labels
        predictions_start, predictions_end = x
        confs_start, preds_start = torch.max(predictions_start, dim=1)
        confs_end, preds_end = torch.max(predictions_end, dim=1)
        
        # convert to numpy tensor
        confs_start, preds_start = confs_start.cpu().numpy(), preds_start.cpu().numpy()
        confs_end, preds_end = confs_end.cpu().numpy(), preds_end.cpu().numpy() 
        
        outputs = list()    
        for i in range(len(preds_start)):
            start = preds_start[i]
            end = preds_end[i]
            ans = self.tokenizer.decode(contexts_ids[i][start:end+1])
            outputs.append(ans)

        return outputs
    
    def forward(self, 
                context: torch.Tensor, 
                question: torch.Tensor, 
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
        
        assert len(context.size()) == 2
        assert len(question.size()) == 2
        
        # set to 2 due to predict the start,end over possibles location in the context
        decoding_sequence = 2 
        
        # encode context and question
        out_context, hidden_context, cell_context = self.context_encoder(context) # n_layers * n_directions(2 for bidirectional), bs, hid_dim
        out_question, hidden_question, cell_question = self.question_encoder(question) # n_layers * n_directions(2 for bidirectional), bs, hid_dim
        end2end_encoding = torch.cat((out_context, out_question), dim=1)
        
        outputs, _ = self.end2end_encoder(end2end_encoding)
        
        # concate the context and question hidden states
        # n_layers*n_directions, b_size, h_dim --> b_size, n_layers*n_directions*h_dim
        # context_hidden = question_hidden.permute(1,0,2).flatten(start_dim=1)
        # question_hidden = question_hidden.permute(1,0,2).flatten(start_dim=1)
        
        # hidden_state_cat = torch.cat((context_hidden, question_hidden), dim=1) # b_size, 2*(n_layers*n_directions*h_dim)
        logits = self.classifier(outputs) # batch_size, max_sequence, num_classes==2
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
        questions_encodings, contexts_encodings, answers_start_positions, answers_end_positions  = self._preprocess_train_batch(batch)
        questions_ids, _ = questions_encodings['input_ids'], questions_encodings['attention_mask']
        contexts_ids, _ = contexts_encodings['input_ids'], contexts_encodings['attention_mask']
        
        # pass forward the model
        start_logits, end_logits = self(context=torch.tensor(questions_ids, dtype=torch.int32, device=self.device), 
                                        question=torch.tensor(contexts_ids, dtype=torch.int32, device=self.device))
        
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


    def validation_step(self, batch, batch_idx):
        pass
 

    def test_step(self, batch):
        pass
                
    def validation_epoch_end(self, outs):
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
        return self.optimizer(self.parameters(), **self.hparams.optimizer_cfg.args)
    
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
from easydict import EasyDict as edict
from argparse import ArgumentParser
import yaml
import time

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers 
from torch.nn.utils.rnn import pad_sequence

from qan.trainer import * 
from qan import datasets
from qan.utils import initialize_dataset, random_seed


def collate_batch(batch):
    label_list, text_list, len_text = [], [], []
    for (_text, _label, _len_text) in batch:
        label_list.append(_label)
        len_text.append(_len_text)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    len_text = torch.tensor(len_text, dtype=torch.int64)

    # pad the text
    text_list = pad_sequence(text_list, batch_first=True)
    return text_list, label_list, len_text   


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', default='', type=str, help='path to trainer config')

    # trainer
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    config = edict(yaml.safe_load(open(args.cfg, 'r')))
    print(f"config: {config}")

    # seed everything
    random_seed(config.seed)
    pl.seed_everything(config.trainer.seed)

    # ------------
    # data
    # ------------
    
    # dataset
    dataset_name = str(config.datasets.name)
    dataset_train = initialize_dataset(datasets, dataset_name, **config.datasets.train)
    vocab = dataset_train.get_vocab()
    config.datasets.test.vocab = vocab
    dataset_test =  initialize_dataset(datasets, dataset_name, **config.datasets.test)
    
    print(f"Length Dataset Train: {len(dataset_train)}")
    print(f"Length Dataset Test: {len(dataset_test)}")
    print(f"Train ratio positive/negative:", dataset_train.get_ratio())
    print(f"Train statistics:", dataset_train.get_review_statistics())
    print("Train len unique words:", len(dataset_train.get_counter()))
    print("Train len vocabulary:", len(vocab))

    # ------------
    # model
    # ------------    
    config.model.args.vocab_size = len(vocab)
    model = TextClassification(model_cfg=config.model,
                               trainer_cfg=config.trainer,
                               vocab=vocab)

    # ------------
    # dataloader
    # ------------   
    train_loader = DataLoader(dataset_train, collate_fn=collate_batch, **config.dataloaders.train)
    test_loader = DataLoader(dataset_test, collate_fn=collate_batch, **config.dataloaders.test)
    
    
    # ------------
    # training
    # ------------
    logger_cfg = config.trainer.logger
    logger = getattr(loggers, logger_cfg.module)
    
    config.trainer.logger = logger(logger_cfg.export_path, name=logger_cfg.experiment_name)

    trainer = pl.Trainer.from_argparse_args(config.trainer)
    
    start = time.time()
    trainer.fit(model, train_loader, test_loader)
    end = time.time()
    print(f"Training time: {end-start} s")
    

if __name__ == '__main__':
    cli_main()
from easydict import EasyDict as edict
from argparse import ArgumentParser
import yaml
import json
import time
import os.path as osp
import os
from tqdm import tqdm

import pytorch_lightning as pl
from qan.utils.initializer import create_instance

  
def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default='', type=str, help='path to trainer config')
    parser.add_argument('-p', '--pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to checkpoint .ckpt for resume training')
    args = parser.parse_args()

    # trainer
    config = edict(yaml.safe_load(open(args.config, 'r')))
    print(f"config: {config}")

    # seed everything
    pl.seed_everything(config.seed)
  
    # init model and dataloader
    dataloader = create_instance(config.dataloader)
    model = create_instance(config.model)
    
    # use pretrained weight 
    if args.pretrained:
        model = model.load_from_checkpoint(config.pretrained)

    logger = create_instance(config.logger)
    
    # resume training
    if args.resume:
        trainer = pl.Trainer(resume_from_checkpoint=config.resume, 
                             **config.trainer)
    else:
        trainer = pl.Trainer(logger=logger, 
                            **config.trainer)
    start = time.time()
    trainer.fit(model, dataloader)
    end = time.time()
    print(f"Training time: {end-start} s")
    
if __name__ == '__main__':
    cli_main()
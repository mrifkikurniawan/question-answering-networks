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
    args = parser.parse_args()

    # trainer
    config = edict(yaml.safe_load(open(args.config, 'r')))
    print(f"config: {config}")

    # seed everything
    pl.seed_everything(config.seed)
  
    dataloader = create_instance(config.dataloader)
    val_dataloader =  dataloader.val_dataloader()
    model = create_instance(config.model)
    logger = create_instance(config.logger)
    trainer = pl.Trainer(logger=logger, 
                         **config.trainer)
    start = time.time()
    trainer.fit(model, dataloader)
    end = time.time()
    print(f"Training time: {end-start} s")
    
    print("Predict on dev dataset")
    all_answers = list()
    ids = list()
    for batch in tqdm(val_dataloader, "Predictions"):
        batch = edict(batch)
        id = batch.id
        ans_list = model.predict(batch)
        all_answers.extend(ans_list)
        ids.extend(id)
        
    # create answer dict
    answers_log = edict()
    for i in range(len(all_answers)):
        answers_log[ids[i]] = all_answers[i]
    
    # save to json file
    if not osp.isdir(logger.log_dir):
        os.makedirs(logger.log_dir)
    json_path = osp.join(logger.log_dir, "predictions_dev.json")
    with open(osp.join(json_path), 'w') as file:
        json.dump(answers_log, file)
        print(f"Save predictions_dev.json to {json_path}")
    
if __name__ == '__main__':
    cli_main()
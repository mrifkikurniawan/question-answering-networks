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
    parser.add_argument('-ckpt', '--checkpoint', default='', type=str, help='path to weight checkpoint')
    parser.add_argument('-o', '--output', default='', type=str, help='path to json output file')
    parser.add_argument('-t', '--threshold', default=None, type=float, help='inference threshold for answerable/not answerable')
    args = parser.parse_args()

    checkpoint = args.checkpoint
    output_path = args.output
    config_path = args.config
    threshold = args.threshold
    
    # trainer
    config = edict(yaml.safe_load(open(config_path, 'r')))
    print(f"config: {config}")
    dataset_name = config.dataloader.args.dataset 

    # seed everything
    pl.seed_everything(config.seed)
  
    dataloader = create_instance(config.dataloader)
    val_dataloader =  dataloader.val_dataloader()
    model = create_instance(config.model).load_from_checkpoint(checkpoint)
    model.eval()
    
    # -----------------
    # Prediction on val set
    # -----------------
    print("Predict on dev dataset")
    all_answers = list()
    ids = list()
    # i = 0
    for batch in tqdm(val_dataloader, "Predictions"):
        batch = edict(batch)
        id = batch.id
        ans_list = model.predict(batch, threshold=threshold)
        all_answers.extend(ans_list)
        ids.extend(id)
        # i +=1
        # if i == 1:
        #     break
        
    # create answer dict
    answers_log = edict()
    for i in range(len(all_answers)):
        answers_log[ids[i]] = all_answers[i]
    
    # save to json file
    if not osp.isdir(output_path):
        os.makedirs(output_path)
    json_path = osp.join(output_path, f"predictions_dev_{dataset_name}.json")
    with open(osp.join(json_path), 'w') as file:
        json.dump(answers_log, file)
        print(f"Save predictions_dev.json to {json_path}")
    
if __name__ == '__main__':
    cli_main()
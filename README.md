
---

<div align="center">    

# Question Answering Nets  
</div>
 
## Description   
Deep learning models trainer for question answering task especially on standard datasets of Squadv1 and Squadv2. At the moment, we have provided transformer-based and RNN-based encoder which the configuration file can be explored in the folder [configs](configs/).

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/mrifkikurniawan/question-answering-networks.git

# install project   
cd question-answering-networks
pip install -e .   
 ```   
 Next, navigate to any file and run it.   
 ```bash
# training seq2seq for QA
# run training module (example: seq2seq on squadv1)   
python3 scripts/train.py --config configs/seq2seq_squadv1_pretrained.yaml 
```

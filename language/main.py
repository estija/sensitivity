# %%
# %%
from datasets import load_dataset
import os
os.getcwd() 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
import pdb
from test import *
warnings.filterwarnings("ignore")
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.optimization import Adafactor, AdafactorSchedule
import torch.backends.cudnn as cudnn
from utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
import logging
import sys
import transformers
import torch
import time
import argparse
from tqdm import tqdm
import string
from RoBERTa import *
from LSTM import *
from GPT2 import *
from transformers import RobertaTokenizer
from transformers import AutoTokenizer
import logging    # first of all import the module
from datetime import datetime

import pandas as pd

parser = argparse.ArgumentParser("main")

parser.add_argument('--train_num_points', type=int,             default = 1000,            help='train data number')
parser.add_argument('--valid_num_points', type=int,             default = 500,               help='validation data number')
parser.add_argument('--report_num_points',type=int,             default = 500,              help='report number')
parser.add_argument('--model_name',       type=str,             default = 'roberta-scratch',   help='model name')
parser.add_argument('--max_length',       type=int,             default=32,                 help='max_length')
parser.add_argument('--num_labels',       type=int,             default=3,                 help='num_labels')
parser.add_argument('--batch_size',       type=int,             default=4,                help='Batch size')
parser.add_argument('--num_workers',      type=int,             default=0,                  help='num_workers')
parser.add_argument('--replace_size',     type=int,             default=3,                  help='to test sensitivity, we need to replance each word by x random words from vocab, here we specify the x')
parser.add_argument('--epochs',           type=int,             default=500,                  help='num of epochs')
parser.add_argument('--lr',               type=float,           default=1e-4,              help='lr')
parser.add_argument('--weight_decay',     type=float,           default=0.0,                 help='weight_decay')
parser.add_argument('--gamma',            type=float,           default=1,                 help='lr*gamma after each test')
parser.add_argument('--vocab_size',       type=int,             default=-1,                help='size of vocabulary')
parser.add_argument('--embedding_dim',    type=int,             default=256,               help='embedding dimension')
parser.add_argument('--hidden_dim',       type=int,             default=256,               help='hidden dimension')
parser.add_argument('--num_head',         type=int,             default=4,               help='num of head in lstm')
parser.add_argument('--num_layers',       type=int,             default=8,                 help='number of layers for LSTM and roberta')
parser.add_argument('--dropout',          type=float,             default=0.2,               help='dropout')
parser.add_argument('--pad_idx',          type=int,             default=1,                 help='ignores token with this index')
parser.add_argument('--sensitivity_method',type=str,            default='none',                 help='embedding/word')
parser.add_argument('--embedding_noise_variance',         type=float,            default=15,                 help='the variance of the noise for the embedding sensitivity testing')
parser.add_argument('--exp_name',                          type=str,            default='default',                 help='why you run this experiment?')
parser.add_argument('--dataset',                          type=str,            default='qqp_p',                 help='data name')
parser.add_argument('--roberta_act',        type=str, default='softmax' , help='softmax/relu')
parser.add_argument('--optimizer',        type=str, default='AdamW' , help='AdamW or SGD')
parser.add_argument('--cudaname',        type=str, default='cuda' , help='whichgpu')
args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb

device = args.cudaname
#foldername = datetime.now().strftime(f'./logs/{args.dataset}/{args.model_name}/%Y_%m_%d_%H_%M_%S')
if 'roberta' in args.model_name and args.roberta_act == 'relu':
    foldername = f'./logs/{args.dataset}/{args.model_name}_{args.roberta_act}/{args.exp_name}'
else:
    foldername = f'./logs/{args.dataset}/{args.model_name}/{args.exp_name}'
# Create the folder if it doesn't exist
if not os.path.exists(foldername):
    os.makedirs(foldername)
# Define the log filename inside the newly created folder
logfilename = os.path.join(foldername, 'logfile.log')

logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)

logging.basicConfig(filename=logfilename, filemode='w', format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s')
handle = "root"
logger = logging.getLogger(handle)
logger.info(f'args:{args}')


if(args.dataset == 'boolq'):
    dataset = load_dataset('boolq')
elif (args.dataset in ['mnli', 'qqp', 'rte', 'qnli', 'mrpc']):
    dataset = load_dataset('glue', args.dataset )
elif(args.dataset == 'imdb'):
    dataset = load_dataset('stanfordnlp/imdb')
    loaded_dataset = pd.read_csv('contrast_data_set/test_original.tsv', sep='\t')
    loaded_dataset['Sentiment'] = loaded_dataset['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
    loaded_validation = {
        'label': loaded_dataset['Sentiment'].tolist(),
        'text': loaded_dataset['Text'].tolist(),
        'dataset':'imdb'
    }
    loaded_dataset = pd.read_csv('contrast_data_set/test_contrast.tsv', sep='\t')
    loaded_dataset['Sentiment'] = loaded_dataset['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
    extra_validation = {
        'label': contrast_data_contrast['Sentiment'].tolist(),
        'text': contrast_data_contrast['Text'].tolist(),
        'dataset':'imdb'
    }
elif(args.dataset == 'qqp_p'):#qqp with perturbation
    dataset = load_dataset('glue', 'qqp' )
    loaded_dataset = pd.read_csv('perturb_qqp_dataset/qqp_same.csv')
    print(loaded_dataset.head())
    loaded_validation = {
        'label': loaded_dataset['label'].tolist(),
        'question1': loaded_dataset['question1'].tolist(),
        'question2': loaded_dataset['question2'].tolist(),
        'dataset':'qqp_p'
    }
    extra_validation = {
        'label': loaded_dataset['label'].tolist(),
        'question1': loaded_dataset['p_question1'].tolist(),
        'question2': loaded_dataset['question2'].tolist(),
        'dataset':'qqp_p'
    }

logger.info('\n Property of dataset:')
logger.info(f'train set size: {len(dataset["train"])}')
# logger.info(f'validation_mismatched set size: {len(dataset["validation_matched"])}')
# logger.info(f'test_matched set size: {len(dataset["test_matched"])}')
# logger.info(f'test_mismatched set size: {len(dataset["test_mismatched"])}')
# %%
# %%

train_num_points = min(args.train_num_points, len(dataset['train']))
train = dataset['train'].shuffle(seed=42).select(range(train_num_points))

if(args.dataset in ['boolq', 'qqp', 'rte', 'qnli', 'mrpc']):
    valid_num_points = min(args.valid_num_points, len(dataset['validation']))
    valid = dataset['validation'][-valid_num_points:]
    valid['dataset'] = args.dataset
    replaced = replaced_data_binary(valid, args.replace_size) 
    replaced['dataset'] = args.dataset
elif (args.dataset == 'mnli'):
    valid_num_points = min(args.valid_num_points, len(dataset['validation_matched']))
    valid = dataset['validation_matched'][-args.valid_num_points:]
    valid['dataset'] = args.dataset
    replaced = replaced_data(valid, args.replace_size) 
    replaced['dataset'] = args.dataset
elif (args.dataset == 'imdb'): #we just test the accuracy on contrast set, valid dataset now is the contrast origin.
    valid_num_points = min(args.valid_num_points, len(dataset['test']))
    if True:#whether we validation on original test dataset or the data we loaded
        valid = loaded_validation
    else:
        shuffled_test = dataset['test'].shuffle(seed=42)
        valid = shuffled_test[-args.valid_num_points:]#
    valid['dataset'] = args.dataset
    extra_data = extra_validation
elif (args.dataset == 'qqp_p'): #we just test the accuracy on contrast set, valid dataset now is the contrast origin.
    valid_num_points = min(args.valid_num_points, len(dataset['test']))
    if True:
        valid = loaded_validation
    else:
        shuffled_test = dataset['test'].shuffle(seed=42)
        valid = shuffled_test[-args.valid_num_points:]#
    valid['dataset'] = args.dataset
    extra_data = extra_validation

# Shuffle the subset and then select the number of points for evaluation
embedding_sens_eval_data = train.shuffle(seed=42).select(range(valid_num_points))[:]
train = train[:]
train['dataset'] = args.dataset
embedding_sens_eval_data['dataset'] = args.dataset


if args.model_name=='roberta-scratch' or args.model_name=='gpt2-scratch':
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
elif args.model_name=='lstm':
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
args.vocab_size = tokenizer.vocab_size
#mnli
#The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. The authors of the benchmark use the standard test set, for which they obtained private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.

# %%
# pdb.set_trace()
contrast_data = None
if(args.dataset in ['boolq', 'rte', 'qqp', 'qnli', 'mrpc','imdb','qqp_p']):
    train_data = get_Dataset_binary(train, tokenizer, args.model_name, max_length=args.max_length)
    train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                            batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
    valid_data = get_Dataset_binary(valid, tokenizer, args.model_name, max_length=args.max_length)
    valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                            batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
    if args.dataset!='imdb' and args.dataset!='qqp_p':
        replaced_data = get_Replaced_Dataset_binary(replaced, tokenizer, args.model_name, max_length = args.max_length)
        replaced_dataloader = DataLoader(replaced_data, sampler=SequentialSampler(replaced_data), 
                                batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
        embedding_sens_eval_data = get_Dataset_binary(embedding_sens_eval_data, tokenizer, args.model_name, max_length = args.max_length)
        embedding_sens_eval_dataloader = DataLoader(embedding_sens_eval_data, sampler=SequentialSampler(replaced_data), 
                                batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
        print("Length of Embedding Sensitivity Eval Loader", len(embedding_sens_eval_dataloader))
    else:
        replaced_dataloader = None
        embedding_sens_eval_dataloader = None
    
elif (args.dataset == 'mnli'):
    train_data = get_Dataset(train, tokenizer, args.model_name, max_length=args.max_length)
    train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                            batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
    valid_data = get_Dataset(valid, tokenizer, args.model_name, max_length=args.max_length)
    valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                            batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
    replaced_data = get_Replaced_Dataset(replaced, tokenizer, args.model_name, max_length = args.max_length)
    replaced_dataloader = DataLoader(replaced_data, sampler=SequentialSampler(replaced_data), 
                            batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
    embedding_sens_eval_data = get_Dataset(embedding_sens_eval_data, tokenizer, args.model_name, max_length = args.max_length)
    embedding_sens_eval_dataloader = DataLoader(embedding_sens_eval_data, sampler=SequentialSampler(replaced_data), 
                            batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
if args.dataset == 'imdb':                            
    extra_data = get_Dataset_binary(extra_data, tokenizer, args.model_name, max_length=args.max_length)
    extra_valid_dataloader = DataLoader(extra_data, sampler=SequentialSampler(extra_data), 
                                batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
if args.dataset == 'qqp_p':                            
    extra_data = get_Dataset_binary(extra_data, tokenizer, args.model_name, max_length=args.max_length)
    extra_valid_dataloader = DataLoader(extra_data, sampler=SequentialSampler(extra_data), 
                                batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)

# %%
if args.model_name=='roberta-scratch':
    model = TextClassifier(args,foldername).to(device)
elif args.model_name=='roberta-base':
    model = TextClassifier(args,foldername).to(device)
elif args.model_name=='gpt2-scratch':
    model = GPT2TextClassifier(args,foldername).to(device)
elif args.model_name=='lstm':
    model = LSTMTextClassifier(args,foldername).to(device)
model.train(train_dataloader,valid_dataloader,embedding_sens_eval_dataloader,replaced_dataloader,device,extra_valid_dataloader=extra_valid_dataloader)

sizeinmb = get_model_size(model)
logger.info(f'model size: {sizeinmb}MB')



# %%






# %%




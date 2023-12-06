import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import csv
import sys
import tqdm
import torch
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
import pandas as pd
import transformers
# transformers.logging.set_verbosity_info()

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    EarlyStoppingCallback, 
    DebertaV2Tokenizer,
#    WandbCallback,
)

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig

# code_path = "/codebase_path/mistral-src"  # codebase
code_path = "/home/dosisiddhesh/MISTRAL_EXP/Github_repo/mistral-src"
# data_path = Path("/dataset_path/Symptom2Disease.csv")  # dataset downloaded from Kaggle
# data_path = Path("/home/dosisiddhesh/MISTRAL_EXP/Github_repo/mistral-src/tutorials/Symptom2Disease.csv")  # dataset downloaded from Kaggle
data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/abstract.csv"

# model_path = Path("/model_path/")  # model and tokenizer location
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/Github_repo/mistral-src/mistral-7B-v0.1")  # model and tokenizer location
tokenizer_path = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model" # sentencepiece tokenizer path
# tokenizer_path = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizers_1.0_50000.json" # bpe tokenizer path
sys.path.append(code_path)  # append the path where mistral-src was cloned

from mistral.tokenizer import Tokenizer
from mistral.model import Transformer, ModelArgs, MyModel


# from ppl_bleu_score_calculator import My_Metric

import torch
from datasets import Dataset
import time

# import wandb
# wandb.login()
# os.environ["WANDB_PROJECT"]="Gal_exp_4"

def model_size_and_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    model_size = sum(t.numel() for t in model.parameters())
    print(f"bert-base-uncased size: {model_size/1000**2:.1f}M parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class HyperParams:
    def __init__(self, epoch = 1, learning_rate = 3e-4, model_id = "facebook/galactica-125m"):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.model_id = model_id
        self.weight_decay=0.1  
        self.warmup_steps=200
        self.lr_scheduler_type="linear"
    #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']

hp = HyperParams(epoch=4, learning_rate=6e-4, model_id="mistral/dummy")

# WANDB_PROJECT=amazon_sentiment_analysis"
# device_ids=[1]
model_id = "mistral"
# model_name = "gal_125m_10ep_6e-4lr_dummy"
model_name = f"{model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_{hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"
# model_dir = os.path.join("/home/dosisiddhesh/GALACTICA_EXP/models", model_name)
model_dir = os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model", model_name)
# wandb_run_name = "continue_pre_training_gal125M_100k_with_lr_estop_dummy"

    

    


# Define training arguments
training_args = TrainingArguments(
    remove_unused_columns=False,
    output_dir=model_dir,  # Change to your desired output directory
    overwrite_output_dir=True,
    per_device_train_batch_size=2,  # Adjust as needed
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=50, # Adjust as needed1
    logging_steps=50,  # Adjust as needed
    # gradient_accumulation_steps=4,
    num_train_epochs=hp.epochs,  # Adjust as needed
    weight_decay=hp.weight_decay,
    warmup_steps=hp.warmup_steps,
    lr_scheduler_type=hp.lr_scheduler_type,
    learning_rate=hp.learning_rate,
    load_best_model_at_end=True, 
    save_steps=200,  # Adjust as needed
    # fp16=True,
    save_total_limit=1,  # Adjust as needed
    logging_dir="./logs_2",
    # report_to="wandb",
    run_name = model_name,
)


# # start a new wandb run to track this script
# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project="continue_pre_training_gal125M",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": learning_rate,
#     "architecture": "Galactica",
#     "dataset": "Arxiv",
#     "epochs": epochs,
#     }
# )

# Load the tokenizer and model
print("Loading tokenizer and model...")

demb,vocab,d_head,d_FF,n_layer,n_head,kv_heads,Window = 4096,50000,128,14336,2,32,8,8192
sequences = []
# demb,  n_layer, d_head,d_FF,n_head,kv_heads,Window,vocab = "1024	32	128	4336	32	8	1024	32000".split()
args = ModelArgs(
    dim=int(demb),
    n_layers=int(n_layer),
    head_dim=int(d_head),
    hidden_dim=int(d_FF),
    n_heads=int(n_head),
    n_kv_heads=int(kv_heads),
    sliding_window=int(Window),
    norm_eps=1e-5,
    vocab_size=int(vocab),
    max_batch_size=1,
)

model = Transformer(args).to("cuda", dtype=torch.float32)
model_size_and_parameters(model)

config = PretrainedConfig(vocab_size=50000, hidden_size=128)
model_tx = MyModel(config, model)


# print the cuda memory usage
print(torch.cuda.memory_summary(device=None, abbreviated=False))

print("Loading tokenizer")
mistral_tokenizer = Tokenizer(tokenizer_path)

# tokenizer_deberta = DebertaV2Tokenizer(
#     vocab_file  = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model",
#     # max_len = 512,
# )
# tokenizer_deberta.save_pretrained('/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_hf.model')


tokenizer = DebertaV2Tokenizer.from_pretrained('/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_hf.model')

print("Tokenizer loaded")
##############_______________ OLD CODE _______________#####################

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# model = model.to('cuda:0')


# model = torch.nn.DataParallel(model, device_ids=[1])
# print(f'Model loaded on devices: {model.device_ids}')


# model_size = sum(t.numel() for t in model.parameters())
# print(f"LLama-2 size: {model_size/1000**3:.1f}B parameters")
#_________________________________________________________________________

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token
##################_______________ OLD CODE _______________#####################
# no need to add special tokens as they are already added in the tokenizer
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#_________________________________________________________________________

def tokenize_function(examples):
    return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
                # return_length=True,                
        )#.to('cuda:0')
    
    # # tokens = tokenizer.encode(examples['text'], bos=True)
    # # tensor = torch.tensor(tokens).to(model.device)
    # token_list = [tokenizer.encode(sample, bos=True) for sample in examples['text']]
    # tensors = torch.tensor(token_list).to(model.device) 

    # tensors = [torch.tensor(tokenizer.encode(sample, bos=True)).to(model.device) for sample in examples['text']]
    # return tensors

print("Loading train dataset...")
# Load your pretraining data
df = pd.read_csv(data_path, nrows=1000)
print(df.head()) 
# print("Loading val dataset...")
# df_eval = pd.read_csv("/home/dosisiddhesh/GALACTICA_EXP/data/arxiv_val.csv", nrows=100)
print("Dataset loaded")

# randomly select 10% of training data as validation data and rest as training data and fix the seed
df_eval = df.sample(frac=0.1, random_state=42)
df = df.drop(df_eval.index) 



# Convert the dataset to a HuggingFace Dataset object
print("Converting dataset to HuggingFace Dataset object...")
train_dataset = Dataset.from_pandas(df)
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
print(train_dataset)
print(f'len(train_dataset) {len(train_dataset)}')


val_dataset = Dataset.from_pandas(df_eval)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

# model.resize_token_embeddings(len(tokenizer))
print('model.resize_token_embeddings(len(tokenizer))')
# model.resize_token_embeddings(len(tokenizer))


print(f'Training arguments: {training_args}')

early_stop = EarlyStoppingCallback(early_stopping_patience=3)

# _________________________________________________________________________________________
# In[]: Trainning the model *****************************************************************
from evaluate import load

metric = load("perplexity")

from transformers import EvalPrediction

def my_compute_metrics(p: EvalPrediction):
    # this function should return 
    # a dictionary {metric_name: score} where score is a float
    # the metrices are perplexity, bleu score and loss value

    # cal ppl
    predictions = p.predictions
    references = p.label_ids
    ppl = metric.compute(predictions=predictions, references=references)

    # cal bleu
    

    # cal loss
    loss = p.loss
    return {"perplexity": ppl, "loss": loss}



# Initialize the Trainer
trainer = Trainer(
    model=model_tx,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # callbacks=[early_stop]#, WandbCallback()]
    # compute_metrics=None,
    compute_metrics=my_compute_metrics,

)

# Train the model
start_time = time.time()
trainer.train()
print(f'Training time: {time.time() - start_time} seconds')

# Save the trained model
print("Saving model...")
trainer.save_model()
print("Model saved")


# trainer.evaluate()
# You can also evaluate the model on a validation dataset if available


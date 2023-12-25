import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import csv
import sys
import time
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler


import numpy as np
import pandas as pd
from pathlib import Path
from evaluate import load
from datasets import Dataset
from prettytable import PrettyTable

metric = load("perplexity")

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    EarlyStoppingCallback, 
    DebertaV2Tokenizer,
    EvalPrediction,
#    WandbCallback,
)


code_path = "/home/dosisiddhesh/MISTRAL_EXP/mistral-src"
data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/abstract.csv"
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/model/mistral-7B-v0.1")  # model and tokenizer location
tokenizer_path = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model" # sentencepiece tokenizer path
sys.path.append(code_path)  # append the path where mistral-src was cloned

from mistral.tokenizer import Tokenizer
from mistral.model import Transformer, ModelArgs, MyModel


# import wandb
# wandb.login()
# os.environ["WANDB_PROJECT"]="Gal_exp_4"
# WANDB_PROJECT=amazon_sentiment_analysis"
# wandb_run_name = "continue_pre_training_gal125M_100k_with_lr_estop_dummy"

def model_size_and_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    model_size = sum(t.numel() for t in model.parameters())
    print(f"MISTRAL model size: {model_size/1000**2:.1f}M parameters")
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

# device_ids=[1]
model_id = "mistral"
model_name = f"{model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_{hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"
model_dir = os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model", model_name)

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


print("Loading tokenizer and model...")
demb,vocab,d_head,d_FF,n_layer,n_head,kv_heads,Window = 4096,50000,128,14336,2,32,8,8192
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
# df_eval = pd.read_csv("/home/dosisiddhesh/GALACTICA_EXP/data/arxiv_val.csv", nrows=100)
print("Dataset loaded")
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

# print('model.resize_token_embeddings(len(tokenizer))')
# model.resize_token_embeddings(len(tokenizer))

early_stop = EarlyStoppingCallback(early_stopping_patience=3)

# _________________________________________________________________________________________
# In[]: Trainning the model *****************************************************************



def my_compute_metrics(p: EvalPrediction):
    # this function should return 
    # a dictionary {metric_name: score} where score is a float
    # the metrices are perplexity, bleu score and loss value
    # cal bleu ?
    # cal ppl
    # cal loss
    return {"perplexity": metric.compute(predictions=p.predictions, references=p.label_ids), "loss": p.loss}


#############_________________________________________ Training the model ________________________________#####################
# We reload the Kaggle dataset from the disk. Each sample is composed of a sentence and a label. The dataset contains 24 different labels.
# data = []  # list of (text, label)
# with open(data_path, newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for i, row in enumerate(reader):
#         if i == 0:  # skip csv header
#             continue
#         data.append((row[2], row[1]))

# # label text to label ID
# labels = sorted({x[1] for x in data})
# txt_to_label = {x: i for i, x in enumerate(labels)}
# print(f"Reloaded {len(data)} samples with {len(labels)} labels.")

# # integer class for each datapoint
# data_class = [txt_to_label[x[1]] for x in data]

''' 
Reloaded 1200 samples with 24 labels.
The task is to classify a symptom, for instance 
`"I have a dry cough that never stops."` to one of the 24 disease labels 
(`Acne, Arthritis, Bronchial Asthma, Cervical spondylosis, Chicken pox, Common Cold, etc.`).
We will now learn a linear classifier on frozen features provided by Mistral 7B.
In particular, each sentence in the dataset will be tokenized, and provided to the model.
If the input sentence is composed of `N` tokens, the model output will be a list of `N` vectors of dimension `d=4096`, where `d` is the dimensionality of the model.
These vectors are then averaged along the dimension 0 to get a vector of size `d`.
Finally, the vectors of all sentences in the dataset are concatenated into a matrix of shape `(D, d)` where `D` is the number of samples in the dataset (in particular, D=1200).
'''

# with torch.no_grad():
#     featurized_x = []
#     # compute an embedding for each sentence
#     for i, (x, y) in tqdm.tqdm(enumerate(data)):
#         tokens = tokenizer.encode(x, bos=True)
#         tensor = torch.tensor(tokens).to(model.device)
#         features = model.forward_partial(tensor, [len(tokens)])  # (n_tokens, model_dim)
#         logprobs = torch.log_softmax(model.forward(tensor, [len(tokens)]), dim=-1)  # (n_tokens, vocab_size)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
num_epochs = 1

# Assuming you have your DataLoader ready (train_loader)
# Assuming you have defined a suitable loss function (criterion)

# Model instantiation
model = Transformer(args)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids, seqlens, labels = batch

        optimizer.zero_grad()
        output = model(input_ids, seqlens)

        # Assuming labels are indices for CrossEntropyLoss
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')

# Optionally, save the trained model
torch.save(model.state_dict(), 'transformer_model.pth')






# # Initialize the Trainer
# trainer = Trainer(
#     model=model_tx,
#     tokenizer=tokenizer,
#     args=training_args,
#     data_collator=data_collator, 
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     # callbacks=[early_stop]#, WandbCallback()]
#     # compute_metrics=None,
#     compute_metrics=my_compute_metrics,

# )

# # Train the model
# start_time = time.time()
# trainer.train()
# print(f'Training time: {time.time() - start_time} seconds')

# # Save the trained model
# print("Saving model...")
# trainer.save_model()
# print("Model saved")

# trainer.evaluate()
# You can also evaluate the model on a validation dataset if available


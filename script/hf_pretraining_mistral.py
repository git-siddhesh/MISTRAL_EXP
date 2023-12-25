import os
device = '1'
os.environ["CUDA_VISIBLE_DEVICES"]=device

import sys
import time
import tqdm
import torch
from pathlib import Path
from evaluate import load
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    EarlyStoppingCallback, 
#    WandbCallback,
)
os.environ['WANDB_DISABLED'] = 'true'
isf16 = False


# metric = load("perplexity")
code_path = "/home/dosisiddhesh/MISTRAL_EXP/mistral-src"
data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/latex.csv"
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/model/mistral-7B-v0.1")  # model and tokenizer location
# tokenizer_path_sentence_piece_for_mistral_src = '/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model'
# tokenizer_path_hf_debertv2 = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_hf.model"
# tokenizer_path_llama = "hf-internal-testing/llama-tokenizer" #llama
tokenizer_path_hf_our = '/home/dosisiddhesh/MISTRAL_EXP/model/hf_tokenizer_2.0%_50000_new'


sys.path.append(code_path)  # append the path where mistral-src was cloned
from mistral.tokenizer import Tokenizer
from mistral.model import Transformer, ModelArgs
from training_utils import Parameter, MyModel, Dataset_Preprocessing, HyperParams
#__________________________________________________________________________________________________
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
D_emb = 4096,
Vocal = 50000,
d_head = 128,
d_FF = 14336,
N_Layer = 2,
N_Head = 16,
KV_Head = 8,,
Window = 8192,
value = [D_emb,Vocal,d_head,d_FF,N_Layer,N_Head,KV_Head,Window]
#**************************************************************************************************
param = Parameter("Mistral", value)
hp = HyperParams(
    epoch=4, 
    learning_rate=6e-4, 
    model_id="mistral/dummy",
    weight_decay=0.1,  
    warmup_steps=200,
    lr_scheduler_type="linear", #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
    BATCH_SIZE=2,
    tokenizer_batch_size=2,
    eval_steps=50, # Adjust as needed1
    logging_steps=50,  # Adjust as needed
    save_steps=200,
    save_total_limit = 1,
)
#__________________________________________________________________________________________________
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


# import wandb
# wandb.login()
# os.environ["WANDB_PROJECT"]="Misral"
# WANDB_PROJECT="Misral_sci_tex"
# wandb_run_name = "dummy"

#____________________________________________________________________________________________________________________________
# In[]: GPU stats ***********************************************************************************************************
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(device))
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# ___________________________________________________________________________________________________________________________
# In[]: preparing the dataset ***********************************************************************************************
dataset_obj = Dataset_Preprocessing(data_path)
print("Loading tokenizer")
# tokenizer = dataset_obj.load_tokenizer(tok_type="mistral_src", tokenizer_path=tokenizer_path_sentence_piece_for_mistral_src)
#-----------------------------------------------------------------------------------------------------------------------------
# if not os.path.exists(tokenizer_path_hf_debertv2):
#     tokenizer_deberta = DebertaV2Tokenizer(
#         vocab_file  = '/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model',
#         # max_len = 512,
#     )
#     tokenizer_deberta.save_pretrained(tokenizer_path)
# tokenizer = dataset_obj.load_tokenizer(tok_type="debertaV2", tokenizer_path=tokenizer_path_hf_debertv2)
#-----------------------------------------------------------------------------------------------------------------------------
tokenizer = dataset_obj.load_tokenizer(tok_type="hf", tokenizer_path=tokenizer_path_hf_our)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#-----------------------------------------------------------------------------------------------------------------------------
print("Loading and preparing dataset...")
dataset_obj.generate_dataset(rows=200, eval_frac=0.1)
#-----------------------------------------------------------------------------------------------------------------------------
print("Loading model...")
model_obj = MyModel(model_id=hp.model_id, hp=hp)
config = model_obj.get_model_config(param)
model = model_obj.get_model(param).to("cuda:0", dtype= torch.float32)
print("Total Params:",model_obj.model_size_and_parameters())
# print('model.resize_token_embeddings(len(tokenizer))')
# model.resize_token_embeddings(len(tokenizer))

# ___________________________________________________________________________________________________________________________
# In[]: Trainning the model *************************************************************************************************

early_stop = EarlyStoppingCallback(early_stopping_patience=3)
# Define training arguments
training_args = TrainingArguments(
    remove_unused_columns=True,
    output_dir=os.path.join("/home/dosisiddhesh/MISTRAL_EXP/model", model_obj.model_name),  # Change to your desired output directory
    overwrite_output_dir=True,
    per_device_train_batch_size=hp.BATCH_SIZE,  # Adjust as needed
    # per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=hp.eval_steps, # Adjust as needed1
    logging_steps=hp.logging_steps,  # Adjust as needed
    # gradient_accumulation_steps=4,
    num_train_epochs=hp.epochs,  # Adjust as needed
    weight_decay=hp.weight_decay,
    warmup_steps=hp.warmup_steps,
    lr_scheduler_type=hp.lr_scheduler_type,
    learning_rate=hp.learning_rate,
    # load_best_model_at_end=True, 
    save_steps=hp.save_steps,  # Adjust as needed
    # fp16=True if isf16 else False,
    # optim='adafactor',
    save_total_limit=hp.save_total_limit,  # Adjust as needed
    logging_dir="./logs_2",
    # report_to="wandb",
    # run_name = model_obj.model_name,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), 
    train_dataset=dataset_obj.get_train_dataset(),
    eval_dataset=dataset_obj.get_val_dataset(),
    # callbacks=[early_stop]#, WandbCallback()],
    # compute_metrics=None,
    # compute_metrics=my_compute_metrics,
)
torch.cuda.empty_cache()
print("GPU untilization before training:")
a,b = torch.cuda.mem_get_info()
gpu_mem_usage = (b-a)/(2**20)
print(f"GPU memory usage: {gpu_mem_usage:.2f} MB")
print_gpu_utilization()

# Train the model
start_time = time.time()
trainer.train()
print(f'Training time: {time.time() - start_time} seconds')
print("Saving model...")
trainer.save_model()
print("Model saved")
trainer.evaluate()
print("Evaluation done")

#____________________________________________________________________________________________________________________________
# In[]: Evaluation **********************************************************************************************************
# def my_compute_metrics(p: EvalPrediction):
#     # this function should return 
#     # a dictionary {metric_name: score} where score is a float
#     # the metrices are perplexity, bleu score and loss value
#     # cal bleu ?
#     # cal ppl
#     # cal loss
#     return {"perplexity": metric.compute(predictions=p.predictions, references=p.label_ids), "loss": p.loss}

#___________________________________________________________________________________________________________________________
# ------------ saving the tokenizer as HF tokenizer -----------------------------------------------------------------------
# tokenizer_deberta = DebertaV2Tokenizer(
#     vocab_file  = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model",
#     # max_len = 512,
# )
# tokenizer_deberta.save_pretrained('/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_hf.model')
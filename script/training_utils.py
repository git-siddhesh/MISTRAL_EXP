
import sys
import pandas as pd
from pathlib import Path
from datasets import Dataset
from prettytable import PrettyTable
from transformers import (
    AutoTokenizer, 
    DebertaV2Tokenizer,
    MistralConfig,
    MistralForCausalLM,
    LlamaTokenizerFast,
)

code_path = "/home/dosisiddhesh/MISTRAL_EXP/mistral-src"
data_path = "/home/dosisiddhesh/MISTRAL_EXP/data/latex.csv"
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/model/mistral-7B-v0.1")  # model and tokenizer location
tokenizer_path = "/home/dosisiddhesh/MISTRAL_EXP/model/tokenizer_5.0%_50000_new.model" # sentencepiece tokenizer path
sys.path.append(code_path)  # append the path where mistral-src was cloned
from mistral.tokenizer import Tokenizer
from mistral.model import Transformer, ModelArgs

class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.D_emb,self.Vocal,self.d_head,self.d_FF,self.N_Layer,self.N_Head,self.KV_Head,self.Window = value

class HyperParams:
    def __init__(self, epoch = 1, learning_rate = 3e-4, model_id = "mistral", **kwargs):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.model_id = model_id
        self.weight_decay=kwargs.get('weight_decay',0.1)  
        self.warmup_steps=kwargs.get('warmup_steps', 200)
        self.lr_scheduler_type=kwargs.get('lr_scheduler_type', "linear") #['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
        self.BATCH_SIZE=kwargs.get('BATCH_SIZE', 2)
        self.tokenizer_batch_size=kwargs.get('tokenizer_batch_size', 2)
        self.eval_steps=kwargs.get('eval_steps', 50) # Adjust as needed1
        self.logging_steps=kwargs.get('logging_steps', 50)  # Adjust as needed
        self.save_steps=kwargs.get('save_steps', 200)
        self.save_total_limit =kwargs.get('save_total_limit', 1)
        self.eval_batch_size=kwargs.get('eval_batch_size',2) #2
        self.eval_frac=kwargs.get('eval_frac', 0.1)
        self.max_seq_length=kwargs.get('max_seq_length', 1024)


    
class MyModel(Parameter, HyperParams):
    def __init__(self, model_id="mistral", hp=None):
        self.model_id = model_id
        self.args = None
        if hp is not None:
            self.model_name = f"{self.model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_\
            {hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"
    
    def get_model_name(self,hp):
        self.model_name = f"{self.model_id}_ep_{hp.epochs}_lr_{hp.learning_rate}_\
            {hp.lr_scheduler_type}_weight_decay_{hp.weight_decay}_warmup_steps_{hp.warmup_steps}"

    # mistral model config from huggingface 
    def get_model_config(self, param):
        self.custom_config = MistralConfig(
            vocab_size=param.Vocal,
            hidden_size=param.D_emb,
            intermediate_size=param.d_FF,
            num_hidden_layers=param.N_Layer,
            num_attention_heads=param.N_Head,
            num_key_value_heads=param.KV_Head,
            hidden_act="silu",
            max_position_embeddings=4096 * 32,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            sliding_window=param.Window,
            attention_dropout=0.0,
        )
        return self.custom_config
    

    
    # mistral model config from mistral-src (github provided)
    def mistral_model_args(self, param):
        self.args = ModelArgs(
            dim = param.D_emb,
            n_layers = param.N_Layer,
            head_dim = param.d_head,
            hidden_dim = param.d_FF,
            n_heads = param.N_Head,
            n_kv_heads = param.KV_Head,
            sliding_window = param.Window,
            norm_eps = 1e-5,
            vocab_size = param.Vocal,
            max_batch_size = 1,
        )
        return self.args 
    
    
    # return the model based on the model type
    # if mistral_src == True, then return the model from mistral-src
    # else return the model from huggingface using MistralForCausalLM
    def get_model(self, mistral_src = False):
        if mistral_src == True:
            print("Loading mistral model from mistral-src")
            self.model = Transformer(self.args)
        else:
            self.model = MistralForCausalLM(self.custom_config)
        return self.model
    
    def model_size_and_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"MISTRAL model size: {model_size/1000**2:.1f}M parameters")
        self.total_params = 0
        self.one_layer_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            self.total_params += params
            if "layers.0" in name:
                self.one_layer_params += params
        print(table)
        print(f"Total Trainable Params: {self.total_params/10**6:.4f}M")
        print(f"Total Trainable Params in one layer: {self.one_layer_params/10**6:.4f}M")
        return self.total_params, self.one_layer_params
    
    

class Dataset_Preprocessing:
    def __init__(self, data_path="/home/dosisiddhesh/MISTRAL_EXP/data/abstract.csv", dataset_batch_size=2, max_seq_length=1024):
        self.dataset_batch_size = dataset_batch_size
        self.max_seq_length = max_seq_length
        self.data_path = data_path

    def load_tokenizer(self, tok_type ,tokenizer_path):
        if tok_type=="mistral_src":
            self.tokenizer = Tokenizer(tokenizer_path)  # mistral-src tokenizer 
        elif tok_type == "debertaV2":
            self.tokenizer =  DebertaV2Tokenizer.from_pretrained(tokenizer_path)  # HF tokenizer
        elif tok_type == "llama":
            self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path) # llama tokenizer
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif tok_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                'cls_token': '<cls>',
                                                'sep_token': '<sep>',
                                                'mask_token': '<mask>',
                                                'unk_token': '<unk>',
                                                'bos_token': '<bos>',
                                                'eos_token': '<eos>'
                                            })
        return self.tokenizer

    def tokenize_function(self, examples):
        return self.tokenizer(                    
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.max_seq_length, #8*1024,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    # return_length=True,
            )#.to('cuda:0')

    def convert_to_hf_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        # return dataset.map(self.tokenize_function, batched=True, remove_columns=dataset.column_names)
        my_dataset = dataset.map(self.tokenize_function, 
                           batched=True, 
                           remove_columns=dataset.column_names, 
                           batch_size=self.dataset_batch_size, 
                           num_proc=8)

        return my_dataset
    

    def generate_dataset(self, row_percent=None, eval_frac=0.1):
        #Setting initial value of the counter to zero 
        rowcount  = 0
        #iterating through the whole file 
        for row in open(self.data_path): 
            rowcount+= 1
        #printing the result 
        print("Number of lines present:-", rowcount)
        row_to_read = rowcount if row_percent is None else int(rowcount * row_percent / 100)
        print("loading sample dataset of size ", row_to_read)
        dataframe = pd.read_csv(self.data_path, nrows=row_to_read)
        dataframe = dataframe.dropna()
        print("Total number of rows after dropping NaN: ", len(dataframe))
        df_eval = dataframe.sample(frac=eval_frac, random_state=42)
        dataframe = dataframe.drop(df_eval.index) 
        print("size of dataframe in MB: ", sys.getsizeof(dataframe)/1000000)
        print("Train dataset size: ", len(dataframe))
        print("Validation dataset size: ", len(df_eval))
        print("Train dataset columns: ", dataframe.columns)
        print("Validation dataset columns: ", df_eval.columns)
        self.train_dataset = self.convert_to_hf_dataset(dataframe)
        self.val_dataset = self.convert_to_hf_dataset(df_eval)
        del dataframe, df_eval

    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset


#  rsync -ra -e 'ssh -p 2020' --info=progress2  SID_DATA_PROCESSED dosisiddhesh@10.0.62.205:/home/dosisiddhesh/
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



from pathlib import Path

# code_path = "/codebase_path/mistral-src"  # codebase
code_path = "/home/dosisiddhesh/MISTRAL_EXP/Github_repo/mistral-src"
# data_path = Path("/dataset_path/Symptom2Disease.csv")  # dataset downloaded from Kaggle
data_path = Path("/home/dosisiddhesh/MISTRAL_EXP/Github_repo/mistral-src/tutorials/Symptom2Disease.csv")  # dataset downloaded from Kaggle
# model_path = Path("/model_path/")  # model and tokenizer location
model_path = Path("/home/dosisiddhesh/MISTRAL_EXP/Github_repo/mistral-src/mistral-7B-v0.1")  # model and tokenizer location

# tokenizer_path = Path()

import csv
import tqdm
import torch
import numpy as np

import sys
sys.path.append(code_path)  # append the path where mistral-src was cloned

from mistral.model import Transformer, ModelArgs
from mistral.tokenizer import Tokenizer

from prettytable import PrettyTable


def model_size_and_parameters(model):
    # Create a PrettyTable for displaying module-wise parameter information
    table = PrettyTable(["Modules", "Parameters"])

    # Calculate the total number of parameters in the model
    model_size = sum(t.numel() for t in model.parameters())

    # Print the total size of the model in megabytes
    print(f"bert-base-uncased size: {model_size/1000**2:.1f}M parameters")

    # Initialize a variable to keep track of the total trainable parameters
    total_params = 0

    # Iterate through named parameters of the model
    for name, parameter in model.named_parameters():
        # Check if the parameter requires gradient (i.e., is trainable)
        if not parameter.requires_grad:
            continue

        # Get the number of parameters in the current module
        params = parameter.numel()

        # Add a row to the PrettyTable with module name and number of parameters
        table.add_row([name, params])

        # Increment the total trainable parameters
        total_params += params

    # Print the PrettyTable with module-wise parameter information
    print(table)

    # Print the total number of trainable parameters in the model
    print(f"Total Trainable Params: {total_params}")

    # Return the total number of trainable parameters
    return total_params

demb,vocab,d_head,d_FF,n_layer,n_head,kv_heads,Window = 4096,100000,128,14336,12,32,8,8192
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
# print the cuda memory usage
print(torch.cuda.memory_summary(device=None, abbreviated=False))

print("Loading tokenizer")
tokenizer = Tokenizer(str(model_path / "tokenizer.model"))

for i in range(10000):
    
    print("loading dataset")
    data = []  # list of (text, label)
    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:  # skip csv header
                continue
            data.append((row[2], row[1]))

    # label text to label ID
    labels = sorted({x[1] for x in data})
    txt_to_label = {x: i for i, x in enumerate(labels)}
    print(f"Reloaded {len(data)} samples with {len(labels)} labels.")

    # integer class for each datapoint
    data_class = [txt_to_label[x[1]] for x in data]

    # with torch.no_grad():
        # compute an embedding for each sentence
    for i, (x, y) in tqdm.tqdm(enumerate(data)):
        tokens = tokenizer.encode(x, bos=True)
        tensor = torch.tensor(tokens).to(model.device)
        features = model.forward_partial(tensor, [len(tokens)])  # (n_tokens, model_dim)


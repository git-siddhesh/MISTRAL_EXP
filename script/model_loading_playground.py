import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from typing import List

from mistral.model import ModelArgs, Transformer
from main import generate

from prettytable import PrettyTable

def model_size_and_parameters(model):
    # Create a PrettyTable for displaying module-wise parameter information
    table = PrettyTable(["Modules", "Parameters"])

    # Calculate the total number of parameters in the model
    model_size = sum(t.numel() for t in model.parameters())

    # Print the total size of the model in megabytes
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    return f'{model_size/1000**2:.1f}'

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
    print(f"Total Trainable Params: {total_params} or {total_params/1000**2:.2f}M parameters")

    # Return the total number of trainable parameters
    return total_params

class DebugTokenizer:
    @property
    def bos_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    @property
    def pad_id(self) -> int:
        return -1

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = [int(x) for x in s.split()]
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return " ".join([str(x) for x in t])

string =  "1024	8 128	4096	32	8	1024	30000"


with open("model_stats2.csv", "a+") as f:
    f.write("ModelParam_M,D_emb,Vocal,D_Head,d_FF,N_Layer,N_Head,KV_Head,Window,GPU_use_MB\n")
    for string in open("model_stats.txt"):
        print(string)
        demb, n_layer, d_head, d_FF, n_head, kv_heads, Window, vocab = string.split()
        try:
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
            params = model_size_and_parameters(model)
            #Returns the global free and total GPU memory for a given device using cudaMemGetInfo.
            a,b = torch.cuda.mem_get_info()
            gpu_mem_usage = (b-a)/(2**20)
            print(f"GPU memory usage: {gpu_mem_usage:.2f} MB")
            f.write(f"{params},{demb},{vocab},{d_head},{d_FF},{n_layer},{n_head},{kv_heads},{Window},{gpu_mem_usage:.2f}\n")

        except Exception as e:
            print(string, e)
            f.write(f"None,{demb},{vocab},{d_head},{d_FF},{n_layer},{n_head},{kv_heads},{Window}, None\n")

        # release all the gpu memory
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()




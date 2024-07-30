import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from ProHabana import main

## HPU Imports
from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
import habana_frameworks.torch.hpu as hthpu  


def generate_text(model, tokenizer, text, device): 
    inputs = tokenizer(text, padding = 'max_length', max_length = 128, truncation = True, return_tensors='pt') 
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model.generate(**inputs) 

    return output 
    


if __name__ == "__main__":
    device = torch.device('hpu') if hthpu.is_available() else torch.device('cpu')
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    text = "Hello World!"
    model = model.to(device)
    generate_text(model, tokenizer, text, device)
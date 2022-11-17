import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np 
import pandas as pd 
import os
import numpy as np
import torch
from tqdm import tqdm
from rake_nltk import Rake
import nltk
import re
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2Model
#from gensim.summarization.summarizer import summarize
from ignite.metrics import Rouge
from ignite.metrics.nlp import Bleu


class StreamlitApp:

    def __init__(self,model_path="./"):

        self.FinalModel = GPT2LMHeadModel.from_pretrained(model_path)
        self.FinalTokenizer = GPT2Tokenizer.from_pretrained(model_path)

    def inference(self,sentence,maxlen=500):

        ids = self.FinalTokenizer.encode(f'{sentence}', return_tensors='pt')

        final_outputs = self.FinalModel.generate(ids,do_sample=True,max_length=maxlen,
                        pad_token_id=self.FinalModel.config.eos_token_id,top_k=50,top_p=0.95,)

        return self.FinalTokenizer.decode(final_outputs[0], skip_special_tokens=True)


    def construct_app(self):

        sequence = st.text_input("Enter Some Text")
        if(len(sequence)>0):
            st.write(self.inference(sequence))


sa = StreamlitApp()
sa.construct_app()
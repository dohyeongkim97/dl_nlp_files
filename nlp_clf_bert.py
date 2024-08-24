#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import transformers
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

import accelerate
from accelerate import Accelerator

import langchain


# In[2]:


from transformers import T5Tokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler


# In[3]:


from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.document_loaders import PDFPlumberLoader, PyMuPDFLoader, PyPDFLoader, UnstructuredPDFLoader

import peft
from peft import PeftModel


# In[4]:


import datasets
from datasets import Dataset
from transformers import Trainer, TrainingArguments


# In[5]:


from transformers import T5ForTokenClassification


# In[6]:


from torch import nn


# In[7]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[8]:


model = T5ForTokenClassification.from_pretrained(
    pretrained_model_name_or_path = 't5-small'
).to(device)


# In[9]:


epochs = 5
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5ForTokenClassification.from_pretrained(
    pretrained_model_name_or_path='t5-small',
)


# In[10]:


data = pd.read_csv("train.csv")


# In[11]:


df = data['first_party'] + data['second_party'] + data['facts'] 


# In[12]:


df = pd.DataFrame(df)
df = pd.concat([df, data['first_party_winner']], axis=1)


# In[13]:


df.columns = ['informations', 'label']


# In[14]:


def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds.cpu().numpy(), axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[15]:


def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text = data.informations.tolist(),
        padding= 'longest',
        truncation = True,
        return_tensors = 'pt'
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    labels = torch.tensor(data.label.values, dtype=torch.long).to(device)
    return TensorDataset(input_ids, attention_mask, labels)


# In[16]:


def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler = data_sampler, batch_size = batch_size)
    return dataloader


# In[17]:


import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch import optim
from transformers import BertForSequenceClassification
from torch import nn
import math
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# In[18]:


epochs = 5
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-multilingual-cased',
    do_lower_case = False
)


# In[19]:


train, valid, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.6*len(df)), int(0.8*len(df))]
)


# In[20]:


train


# In[21]:


valid


# In[22]:


train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)


# In[23]:


model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path='bert-base-multilingual-cased',
    num_labels = 2
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)


# In[25]:


def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0.0
    
    for input_ids, attention_mask, labels in dataloader:
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        
        loss = outputs.loss
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(dataloader)
    return train_loss


# In[26]:


def evaluation(model, dataloader):
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss, val_accuracy = 0.0, 0.0
        
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            logtis = logits.detach().cpu().numpy()
            labels_ids = labels.to('cpu').numpy()
            accuracy = calc_accuracy(logits, labels_ids)
            
            val_loss += loss
            val_accuracy += accuracy
            
        val_loss = val_loss/len(dataloader)
        val_accuracy = val_accuracy / len(dataloader)
        return val_loss, val_accuracy


# In[27]:


best_loss = 10000
for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f'Epoch: {epoch+1} train loss: {train_loss:.4f} val loss: {val_loss:.4f} val_acc: {val_accuracy:.4f}')
    
    if val_loss < best_loss:
        best_loss = val_loss


# In[28]:


test_loss, test_score = evaluation(model, test_dataloader)


# In[29]:


test_loss


# In[30]:


test_score


# In[ ]:





# In[ ]:





# In[ ]:





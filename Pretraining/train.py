
import os

if not (os.path.isfile('/kaggle/working/my_tokenizer.json') and os.path.isfile('/kaggle/working/pretraining_dataset.bin')):
    os.system("gdown 1bLhcaSOg8u34KXt3pblZVFQCG_VySUrO")
    os.system("gdown 1kQZX-fwBr2cIYN4DPqZjoeKWNdh8ym2O")

"""# Imports"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import GradScaler
import math
import time
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
import contextlib
import torch.multiprocessing as mp

"""# DDP setup"""

def ddp_setup(rank,world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)

"""# Global Vars"""

dtype=torch.float16
vocab_size= 32000
embed_dim= 768
n_head= 12
n_layer= 12
device='cuda' if torch.cuda.is_available() else 'cpu'
batch_size= 12
epochs = 1

max_seq_len=1024

weight_decay=0.1
b1=0.9
b2=0.95
peak_lr=3e-4
gradient_accumulation_steps= 4
BASE_DIR='/kaggle/working/'
update_checkpoint_steps=100
update_plot_steps=100

"""# RoPE"""

class RotaryPositionalEmbeddings(nn.Module):

  def __init__(self, d: int, base: int = 10_000):

    super().__init__()
    self.base = base
    self.d = d
    self.cos_cached = None
    self.sin_cached = None
  def _build_cache(self, x: torch.Tensor):
      seq_len = x.shape[1]  # B, T, heads, d -> T is dim 1

      if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
          return

      theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
      seq_idx = torch.arange(seq_len, device=x.device).float()
      idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
      idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

      self.cos_cached = idx_theta2.cos()[None, :, None, :]  # (1, T, 1, d)
      self.sin_cached = idx_theta2.sin()[None, :, None, :]  # (1, T, 1, d)

      d_2 = self.d // 2
      return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

  def forward(self, x: torch.Tensor):
      # x: (B, T, num_heads, head_dim)
      self._build_cache(x)
      neg_half_x = self._neg_half(x)
      x_rope = (x * self.cos_cached[:, :x.shape[1]]) + (neg_half_x * self.sin_cached[:, :x.shape[1]])
      return x_rope

"""# Attention"""

class ScaledDotProductAttention(nn.Module):
  def __init__(self,per_head_embed_dim):
    super().__init__()
    self.softmax=nn.Softmax(dim=-1)
    self.d=per_head_embed_dim

  def forward(self,Q,K,V):
    return nn.functional.scaled_dot_product_attention(Q,K,V,is_causal=True)

"""# Layer"""

class Llm_layer(nn.Module):
  def __init__(self,n_head,embed_dim,dropout_ratio=0.1):
    super().__init__()
    self.embed_dim=embed_dim
    self.n_head=n_head
    self.linear1=nn.Linear(embed_dim,embed_dim*3)
    self.attn_layer=ScaledDotProductAttention(per_head_embed_dim=embed_dim//n_head).to(device)
    self.lyr_norm1=nn.LayerNorm(embed_dim)
    self.linear2=nn.Linear(embed_dim,embed_dim)
    self.lyr_norm2=nn.LayerNorm(embed_dim)
    self.ffn=nn.Sequential(nn.Linear(embed_dim,embed_dim*4),nn.GELU(),nn.Dropout(p=dropout_ratio),nn.Linear(embed_dim*4,embed_dim)).to(device)
    self.rope=RotaryPositionalEmbeddings(d=embed_dim//n_head).to(device)
    self.linear_dropout=nn.Dropout(p=dropout_ratio)
    self.ffn_dropout=nn.Dropout(p=dropout_ratio)


  def forward(self,x):
    x0=self.linear1(self.lyr_norm1(x))
    Q,K,V=x0.split(self.embed_dim,dim=-1)

    Q=self.rope(Q.reshape(Q.shape[0],Q.shape[1],self.n_head,self.embed_dim//self.n_head)).transpose(1,2)
    K=self.rope(K.reshape(K.shape[0],K.shape[1],self.n_head,self.embed_dim//self.n_head)).transpose(1,2)
    V=V.reshape(V.shape[0],V.shape[1],self.n_head,self.embed_dim//self.n_head).transpose(1,2)

    res=self.attn_layer(Q,K,V)
    a,b,c,d=res.shape
    res=res.transpose(1,2).reshape(a,c,self.embed_dim)

    x=x+self.linear_dropout(self.linear2(res))

    x=x+self.ffn_dropout(self.ffn(self.lyr_norm2(x)))
    return x

"""# LLM"""

# @title LLM
class LLM(nn.Module):
  def __init__(self,vocab_size,embed_dim,n_layer,n_head,dropout_ratio=0.1):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size,embed_dim)
    self.embed_dropout = nn.Dropout(p=dropout_ratio)
    self.llm_layers=nn.ModuleList([Llm_layer(n_head=n_head,embed_dim=embed_dim) for i in range(n_layer)])
    self.layer_norm1=nn.LayerNorm(embed_dim)
    self.linear=nn.Linear(embed_dim,vocab_size)
    self.linear.weight=self.embedding.weight
    self.softmax=nn.Softmax(dim=-1)

  def forward(self,x):
    x=self.embed_dropout(self.embedding(x))
    for llm_layer in self.llm_layers:
      x=llm_layer(x)
    x=self.linear(self.layer_norm1(x))

    return x


"""# Custom Dataset"""

class customDatset(Dataset):
  def __init__(self,bin_file_path,max_seq_len):
    self.data = np.memmap(bin_file_path, dtype=np.uint16, mode='r')
    self.max_seq_len=max_seq_len
    self.len=len(self.data)//max_seq_len

  def __getitem__(self,idx):
    arr=self.data[idx*self.max_seq_len:(idx+1)*self.max_seq_len]
    arr=torch.tensor(arr).long()
    return arr

  def __len__(self):
    return self.len


"""# Loss function"""




"""# learning rate scheduler"""

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return current_step / warmup_steps
        # Cosine decay from peak to min_lr
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


"""# Trainer Class"""

class Trainer:
    def __init__(self, gradient_accumulation_steps, epochs,update_plot_steps,update_checkpoint_steps,model,dataset, batch_size,base_dir,gpu_id,max_seq_len=1024):
        self.loss_per_step_lst = []
        self.time_per_step = []
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.update_plot_steps = update_plot_steps
        self.update_checkpoint_steps = update_checkpoint_steps
        self.BASE_DIR = base_dir
        self.start_time = None
        self.gpu_id=gpu_id
        self.model=DDP(model,device_ids=[self.gpu_id])
        self.dataloader=DataLoader(dataset,batch_size=batch_size,sampler=DistributedSampler(dataset),shuffle=False,pin_memory=True)
        self.optimiser = self.create_optimiser()
        self.loss_scaler=GradScaler()
        total_tokens=max_seq_len*len(dataset)*epochs

        tokens_per_step=batch_size*max_seq_len*gradient_accumulation_steps
        total_steps=total_tokens//tokens_per_step

        warmup_steps=int(0.01*total_steps)

        self.lr_scheduler=get_cosine_schedule_with_warmup(self.optimiser,warmup_steps,total_tokens)
        self.steps_per_epoch = total_steps//epochs
        self.cross_entropy_loss=nn.CrossEntropyLoss()
        self.start_epoch=0
        self.resume_training()
        
    def resume_training(self):
        
        checkpoint = torch.load('/kaggle/input/datasets/pankajmaulekhi/checkpointer/checkpoint.pth', map_location='cpu',weights_only=False)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.time_per_step=checkpoint['time_per_step']
        self.loss_per_step_lst=checkpoint['loss_per_step_lst']
        self.start_epoch=checkpoint['epoch']
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
    
    def loss_fn(self,y_pred,y_true):
      return self.cross_entropy_loss(y_pred.reshape(-1,32000),y_true.reshape(-1))
        
    def create_optimiser(self):
        decay_params=[p for n,p in self.model.named_parameters() if p.dim()>=2 ]   # weight matrices only
        no_decay_params=[p for n,p in self.model.named_parameters() if p.dim()<2 ]   # biases, layernorm, etc.


        param_groups = [
                {"params": decay_params,    "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
          ]
        optimiser=torch.optim.AdamW(param_groups,lr=peak_lr,betas=(b1,b2))
        return optimiser

    def create_progress_bar(self, steps):
        return tqdm(total=steps, desc="training")

    def _run_epoch(self, model, dataloader, epoch):
        p_bar = self.create_progress_bar(self.steps_per_epoch)
        for i, batch in enumerate(dataloader):
            x = batch.to(self.gpu_id)
            is_accumulating = (i+1)%self.gradient_accumulation_steps !=0

            with model.no_sync() if is_accumulating else contextlib.nullcontext():
                with torch.autocast(dtype=torch.float16, device_type=device):
                    y_pred = model(x[:, :-1])
                    y_true = x[:, 1:]
                    raw_loss = self.loss_fn(y_pred=y_pred, y_true=y_true)

                self.loss_per_step_lst.append(raw_loss.item())
                self.time_per_step.append(time.time() - self.start_time)

                loss = raw_loss / self.gradient_accumulation_steps
                scaled_loss = self.loss_scaler.scale(loss)
                scaled_loss.backward()

            if not is_accumulating:
                self.loss_scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.loss_scaler.step(self.optimiser)
                self.loss_scaler.update()
                self.lr_scheduler.step()
                self.optimiser.zero_grad()

            if (i + 1) % self.update_plot_steps == 0:
                plt.figure()
                plt.plot(self.loss_per_step_lst)
                plt.title("Loss vs no of training steps")
                plt.xlabel("No of training steps")
                plt.ylabel("Loss")
                plt.savefig('/kaggle/working/loss_vs_steps.png') 
                plt.close()

                plt.figure()
                plt.plot(self.time_per_step, self.loss_per_step_lst)
                plt.title("Loss vs time")
                plt.xlabel("Time in seconds")
                plt.ylabel("Loss")
                plt.savefig('/kaggle/working/loss_vs_time.png') 
                plt.close()
                
            
            if (i + 1) % self.update_checkpoint_steps == 0  and self.gpu_id==0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimiser_state_dict': self.optimiser.state_dict(),
                    'loss_scaler': self.loss_scaler.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss_per_step_lst': self.loss_per_step_lst,
                    'time_per_step': self.time_per_step,
                    'rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                }, self.BASE_DIR + 'checkpoint.pth')

            p_bar.set_postfix(loss=f"{raw_loss.item():.4f}")
            p_bar.update(1)
        p_bar.close()

    def train(self):
        self.start_time = time.time()
        for epoch in range(self.start_epoch,self.epochs):
            self._run_epoch(self.model, self.dataloader, epoch)


def main(rank,world_size):
    ddp_setup(rank, world_size)

    model=LLM(vocab_size=vocab_size,embed_dim=embed_dim,n_layer=n_layer,n_head=n_head).to(rank)
    dataset=customDatset(BASE_DIR+'pretraining_dataset.bin',max_seq_len)
    trainer=Trainer(gradient_accumulation_steps, epochs, update_plot_steps,
                     update_checkpoint_steps,model,dataset,batch_size, BASE_DIR,rank)
    trainer.train()
    dataset=customDatset(BASE_DIR+'pretraining_dataset.bin',max_seq_len)

    destroy_process_group()


if __name__=="__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main,args=(world_size,),nprocs=world_size)

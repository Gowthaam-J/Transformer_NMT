import torch
import torch.nn as nn

import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
class PostionalEmbedding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
      # Corrected the usage of self.eps
      mean=x.mean(-1,keepdim=True)
      std=x.std(-1,keepdim=True,unbiased=False)
      return self.alpha * (x-mean)/(std+self.eps) + self.bias
class FeedForwardBlock(nn.Module):
  def __init__(self, d_model:int, d_ff:int, dropout:float):
    super().__init__()
    #[batch_size, seq_len, d_model] → [batch_size, seq_len, d_ff]
    self.linear_1=nn.Linear(d_model,d_ff)
    self.dropout=nn.Dropout(dropout)
    self.linear_2=nn.Linear(d_ff,d_model)
    self.activation = nn.GELU()
  def forward(self,x):
    return self.linear_2(self.dropout(self.activation(self.linear_1(x))))
class MultiheadAttention(nn.Module):
  def __init__(self, d_model:int, n_head:int, dropout:float):
    super().__init__()
    self.d_model=d_model
    self.n_head=n_head
    assert d_model % n_head==0, "d_model is not deivisible by h"
    self.d_k=d_model//n_head
    self.w_Q=nn.Linear(d_model,d_model)
    self.w_K=nn.Linear(d_model,d_model)
    self.w_V=nn.Linear(d_model,d_model)
    self.w_o=nn.Linear(d_model,d_model)
    self.dropout=nn.Dropout(dropout)

  @staticmethod
  def attention(query,key,value,mask,dropout:nn.Dropout):
    d_k=query.shape[-1]
    attention_scores=(query @key.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    attention_scores=attention_scores.softmax(dim=-1)
    if dropout is not None:
      attention_scores=dropout(attention_scores)
    return (attention_scores @ value),attention_scores


  def forward(self,q,k,v,mask):
    q=self.w_Q(q) #(Batch,seq_len,d_model)-->(Batch,seq_len,d-Model)
    k=self.w_K(k) #(Batch,seq_len,d_model)-->(Batch,seq_len,d-Model)
    v=self.w_V(v) #(Batch,seq_len,d_model)-->(Batch,seq_len,d-Model)

    #(Batch,seq_len,d_model)-->(Batch,seq_len,n_head,d_K)-->(Batch,n_head,Seq_len,d_k)
    q=q.view(q.shape[0],q.shape[1],self.n_head,self.d_k).transpose(1,2)
    k=k.view(k.shape[0],k.shape[1],self.n_head,self.d_k).transpose(1,2)
    v=v.view(k.shape[0],v.shape[1],self.n_head,self.d_k).transpose(1,2)
    x,self.attention_scores=MultiheadAttention.attention(q,k,v,mask,self.dropout)
    #(Batch,n_head,seq_len,d_k)-->(Batch,seq_len,n_head,d_k)-->(Batch,seq_len,d_model)
    x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.n_head*self.d_k)
    # ( Batch,seq_len,d_model) --> (Batch,seq_Len,d_model)
    return self.w_o(x)
class ResidualConnection(nn.Module):
  def __init__(self,dropout:float):
    super().__init__()
    self.dropout=nn.Dropout(dropout)
    self.norm=LayerNormalization()
  def forward(self,x,sublayer):
    return x+self.dropout(sublayer(self.norm(x)))
class EncoderBlock(nn.Module):
  def __init__(self,self_attention_block: MultiheadAttention,feed_forward_block:FeedForwardBlock,dropout:float):
    super().__init__()
    self.self_attention_block=self_attention_block
    self.feed_forward_block=feed_forward_block
    self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self,x,src_mask):
      x=self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,src_mask))
      x=self.residual_connections[1](x,self.feed_forward_block)
      return x

class Encoder(nn.Module):
  def __init__(self,layers:nn.ModuleList):
    super().__init__()
    self.layers=layers
    self.norm=LayerNormalization()

  def forward(self,x,mask):
    for layer in self.layers:
      x=layer(x,mask)
    return self.norm(x)
class DecoderBlock(nn.Module):
  def __init__(self,self_attention_block: MultiheadAttention,cross_attention_block:MultiheadAttention,feed_forward_block:FeedForwardBlock,dropout:float):
    super().__init__()
    self.self_attention_block=self_attention_block
    self.cross_attention_block=cross_attention_block
    self.feed_forward_block=feed_forward_block
    self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

  def forward(self,x,encoder_output,src_mask,tgt_mask):
    x=self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
    x=self.residual_connections[1](x,lambda x : self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
    x=self.residual_connections[2](x,self.feed_forward_block)
    return x
class Decoder(nn.Module):
  def __init__(self,layers:nn.ModuleList):
    super().__init__()
    self.layers=layers
    self.norm=LayerNormalization()
  def forward(self,x,encoder_output,src_mask,tgt_mask):
    for layer in self.layers:
      x=layer(x,encoder_output,src_mask,tgt_mask)
    return self.norm(x)
class ProjectionLayer(nn.Module):
  def __init__(self,d_model:int,vocab_size:int):
    super().__init__()
    self.proj=nn.Linear(d_model,vocab_size)
  def forward(self,x):
    #(Batch,Seq_len,d_model) --> (Batch,seq_len,vocab_size)
    return torch.log_softmax(self.proj(x),dim=-1)
class Transformer(nn.Module):
  def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PostionalEmbedding,tgt_pos:PostionalEmbedding,projection_layer:ProjectionLayer):
    super().__init__()
    self.encoder=encoder
    self.decoder=decoder
    self.src_embed=src_embed
    self.tgt_embed=tgt_embed
    self.src_pos=src_pos
    self.tgt_pos=tgt_pos
    self.projection_layer=projection_layer
  def encode(self,src,src_mask):
    src=self.src_embed(src)
    src=self.src_pos(src)
    return self.encoder(src,src_mask)
  def decode(self,encoder_output,src_mask,tgt,tgt_mask):
    tgt=self.tgt_embed(tgt)
    tgt=self.tgt_pos(tgt)
    return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
  def project(self,x):
    return self.projection_layer(x)


def build_transformer(src_vocab_size:int,tgt_vocab_size:int, src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
  src_embed=InputEmbeddings(d_model,src_vocab_size)
  tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)
  src_pos=PostionalEmbedding(d_model,src_seq_len,dropout)
  tgt_pos=PostionalEmbedding(d_model,tgt_seq_len,dropout)
  encoder_block=[]
  for _ in range(N):
    encoder_self_attention_block=MultiheadAttention(d_model,h,dropout)
    feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
    encoder_block.append(EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout))

  decoder_blocks=[]
  for _ in range(N):
    decoder_self_attention_block=MultiheadAttention(d_model,h,dropout)
    decoder_cross_attention_block=MultiheadAttention(d_model,h,dropout)
    feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
    decoder_blocks.append(DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout))

#Create encoder and decoder
  encoder=Encoder(nn.ModuleList(encoder_block))
  decoder=Decoder(nn.ModuleList(decoder_blocks))
#create projection Layer
  projection_layer=ProjectionLayer(d_model,tgt_vocab_size)
#create transformer
  transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
#Xavier Parameters Intitialtision
  for p in transformer.parameters():
    if p.dim()>1:
      nn.init.xavier_uniform_(p)
  return transformer
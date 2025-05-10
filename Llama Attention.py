import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import  Tuple,Optional

#2.33.35



hiddenSize = 128
numAttentionHeads = 16
numKeyValueHeads = 4
headDim = hiddenSize // numAttentionHeads
maxPositionEmbeddings = 256
ropeTheta = 10000.0
rmsNormEps = 1e-5
attentionBias = False
attentionDropout = 0.0
useQkNorm = True

batchSize = 2
sequenceLength = 10
hiddenStates = torch.randn(batchSize,sequenceLength,hiddenSize)
positionIds = torch.arange(0,sequenceLength).unsqueeze(0).repeat(batchSize,1)

#We make sure that no token can predict more than itself and previous ones.
attentionMask = torch.triu(torch.ones(sequenceLength,sequenceLength) * -torch.inf,diagonal=1)
attentionMask = attentionMask.unsqueeze(0).unsqueeze(0)
attentionMask = attentionMask.expand(batchSize,1,-1,-1)

print("\nConfiguration:")
print(f"Hidden Size: {hiddenSize}")
print(f"# Attention Heads: {numAttentionHeads}")
print(f"# Key Value Heads: {numKeyValueHeads}")
print(f"Head Dimension: {headDim}")

print("\n Sample Input Shapes:")
print(f"Hidden States: {hiddenStates.shape}")
print(f"Position Ids: {positionIds.shape}")
print(f"Attention Mask: {attentionMask.shape}")

# - **Q:** Represents the current token's query.
# - **K:** Represents the keys of all tokens in the sequence (or context).
# - **V:** Represents the values (information) of all tokens.


#Define projection layers
qProj = nn.Linear(hiddenSize,numAttentionHeads * headDim,bias=attentionBias)
kProj = nn.Linear(hiddenSize,numKeyValueHeads * headDim,bias= attentionBias)
vProj = nn.Linear(hiddenSize,numKeyValueHeads * headDim,bias=attentionBias)
oProj = nn.Linear(numAttentionHeads * headDim,hiddenSize,bias=attentionBias)


queryStates = qProj(hiddenStates)
keyStates = kProj(hiddenStates)
valueStates = vProj(hiddenStates)

#Reshape Q,K,V for multi head attention
queryStates = queryStates.view(batchSize,sequenceLength,numAttentionHeads,headDim).transpose(1,2)
keyStates = keyStates.view(batchSize,sequenceLength,numKeyValueHeads,headDim).transpose(1,2)
valueStates = valueStates.view(batchSize,sequenceLength,numKeyValueHeads,headDim).transpose(1,2)


print("Projected Shapes: ")
print(f"Query states: {queryStates.shape}")
print(f"Key states: {keyStates.shape}")
print(f"Value states: {valueStates.shape}")

numKeyValueGroups = numAttentionHeads // numKeyValueHeads
print(f"\nNum Key/Value Groups (Q heads per K/V head): {numKeyValueGroups}")


#Rotational Positional Embedding RoPE



def simpleRopeCalculation(dim,maxSeqLen,base = 10000.0,device=None):

    invFreq = 1.0 / (base ** (torch.arange(0,dim,2,device=device).float() / dim))
    t = torch.arange(maxSeqLen,device=device).type_as(invFreq)
    freqs = newFunc(invFreq,t)

    emb = torch.cat((freqs,freqs),dim =-1)
    freqsCos = emb.cos()
    freqsSin = emb.sin()
    freqsCis = torch.complex(freqsCos,freqsSin)

    return freqsCis


def newFunc(invFreq,t):
    freqs = torch.outer(t,invFreq)
    return freqs


def applyRotaryEmbTorch(
        xq:torch.Tensor,
        xk: torch.Tensor,
        freqsCis: torch.Tensor,
) -> Tuple[torch.Tensor,torch.Tensor]:

    freqsCis = freqsCis.to(xq.device)
    freqsCis = freqsCis[positionIds]
    freqsCis = freqsCis[:,None,:,:]

    xQ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))
    xK = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1],-1,2))

    freqsCisBroadcast = freqsCis[..., :xQ.shape[-1]]

    rotatedXq = xQ * freqsCisBroadcast
    rotatedXk = xK * freqsCisBroadcast

    xqOut = torch.view_as_real(rotatedXq).flatten(3)
    xkOut = torch.view_as_real(rotatedXk).flatten(3)

    return xqOut.type_as(xq),xkOut.type_as(xk)



freqsCis = simpleRopeCalculation(headDim,maxPositionEmbeddings,base = ropeTheta,device=hiddenStates.device)
print(f"Calculated freqsCis Shape: {freqsCis.shape} ")

queryStatesRope,keyStatesRope = applyRotaryEmbTorch(queryStates,keyStates,freqsCis)
print("\nShapes after RoPE:")
print(f"QueryStatesRope: {queryStatesRope.shape}")
print(f"KeyStatesRope:{keyStatesRope.shape}")



class SimpleL2Norm(nn.Module):

    def __init__(self,eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self,x):
        return x * torch.rqsrt(x.pow(2).mean(-1,keepdim=True) + self.eps)


if useQkNorm:
    qkNorm = SimpleL2Norm()
    queryStatesFinal = qkNorm(queryStatesRope)
    keyStatesFinal = qkNorm(keyStatesRope)
    print("\n Applied Q-K Norm")

else:
    queryStatesFinal = queryStatesRope
    keyStatesFinal = keyStatesRope
    print("\nSkipped Q-K Norm")

print("\nShapes before attention score calculation:")
print(f"query states final: {queryStatesFinal.shape}")
print(f"key states final: {keyStatesFinal.shape}")



def RepeatKv(hidden_states: torch.Tensor,n_rep:int) -> torch.Tensor:

    batch,numKeyValueHeads,slen,headDim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:,:,None,:,:].expand(batch,numKeyValueHeads,n_rep,headDim)
    return hidden_states.reshape(batch,numKeyValueHeads * n_rep,slen,headDim)




keyStatesRepeated = RepeatKv(keyStatesFinal,numKeyValueGroups)
valueStatesRepeated = RepeatKv(valueStates,numKeyValueGroups)
print(f"\nShapes after repeating K V for G Q A:")
print(f"Key states repeated: {keyStatesRepeated.shape}")
print(f"Value states repeated: {valueStatesRepeated.shape}")


#Scaling
attenWeights = torch.matmul(queryStatesFinal,keyStatesRepeated.transpose(22,3))
scalingFactor = 1.0 / math.sqrt(headDim)
attenWeights = attenWeights * scalingFactor


#Apply Mask
if attentionMask is not None:
    print(f"\n Applying attention mask with shape: {attentionMask.shape}")
    causalMask = attentionMask[:,:,:,:keyStatesRepeated.shape[-2]]
    attenWeights = attenWeights + causalMask
else:
    print(f"\n No attention mask is applied")



#Soft max
attenWeights = nn.functional.softmax(attenWeights,dim=-1).to(queryStates.dtype)
attenOutput = torch.matmul(attenWeights,valueStatesRepeated)

print(f"\n Attention calculation shapes: ")
print(f" Attention Weights (raw) : {attenWeights.shape}")
print(f"Attention Weights (after softmax): {attenWeights.shape}")
print(f"Attention output: {attenOutput.shape}")


attenOutput = attenOutput.transpose(1,2).contiguous()
attenOutput = attenOutput.view(batchSize,sequenceLength,hiddenSize)
finalAttnOutput = oProj(attenOutput)

print("\n Final Output Shapes: ")
print(f"Attention Output (Reshaped) : {attenOutput.shape}")
print(f"Final Attention Output (Reshaped) : {finalAttnOutput.shape}")



class SimplifiedLlama4Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config['max_position_embeddings']
        self.rope_theta = confi['rope_theta']
        self.attention_bias = config['attention_bias']
        self.use_qk_norm = config['use_qk_norm']

        if(self.head_dim 0 self.num_attenion_heads) != self.hidden_size:
            raise ValueError("Hidden size must be divisible by num_attention_heads")


        self.q_proj = nn.Linear(self.hidden_size,self.num_attention_heads * self.head_dim,bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size,self.num_key_value_heads * self.head_dim,bias = self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size,self.num_key_value_heads * self.head_dim,bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim,self.hidden_size,bias=self.attention_bias)

        self.freqs_cis = simpleRopeCalculation(self.head_dim,self.max_position_embeddings,base = self.rope_theta)

        if self.use_qk_norm:
            self.qk_norm = SimpleL2Norm()



    def forward(self,hidden_states,attention_mask,position_ids):
        batch_size,sequence_length, _ = hidden_states.shape

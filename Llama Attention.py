import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import  Tuple,Optional



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

#2.33.35

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from sympy.physics.quantum.gate import gate_simp

hiddenSize = 128
ffnIntermediateratio = 8 / 3
multipleOf = 32
intermediateSize = int(hiddenSize * ffnIntermediateratio)
intermediateSize = ((intermediateSize + multipleOf - 1) // multipleOf) * multipleOf

hiddenAct = "silu"
rmsNormEps = 1e-5
ffnBias = False


batchSize = 2
sequenceLength = 10
inputToFfnBlock = torch.randn(batchSize,sequenceLength,hiddenSize)

print("Configuration: ")
print(f" Hidden Size: {hiddenSize}")
print(f" Intermeadite Size: {intermediateSize} (Calculated from ratio {ffnIntermediateratio:.2f}"
      f"multiple of {multipleOf}")
print(f"Hidden act: {hiddenAct}")
print(f"Rms Norm eps: {rmsNormEps}")

print("\n Sample Input Shape Before FFN Block Norm:")
print(f"Input to FFN block: {inputToFfnBlock.shape}")


class SimplifiedRMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


    def forward(self,hidden_states):
        inputDType = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1,keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(inputDType)



postAttentionNorm = SimplifiedRMSNorm(hiddenSize,eps = rmsNormEps)
normalizedHiddenStates = postAttentionNorm(inputToFfnBlock)
print("Shape after post attention RMS Norm")
print(f"Normalized hidden stats: {normalizedHiddenStates.shape}")

gateProj = nn.Linear(hiddenSize,intermediateSize,bias=ffnBias)
upProj = nn.Linear(hiddenSize,intermediateSize,bias=ffnBias)
downProj = nn.Linear(intermediateSize,hiddenSize,bias=ffnBias)

if hiddenAct == "silu":
    activationFn = nn.SiLU()

else:
    raise NotImplementedError(f"Activation {hiddenAct} not implemented in this example")


gateOutput = gateProj(normalizedHiddenStates)
upOutput = upProj(normalizedHiddenStates)
activatedGate = activationFn(gateOutput)
gatedResult = activatedGate * upOutput
ffnOutput = downProj(gatedResult)

print("\n Shapes within FFN: ")
print(f" Gate output: {gateOutput.shape}")
print(f" Up output: {upOutput.shape}")
print(f" Gated Result: {gatedResult.shape}")
print(f"FFN Output: { ffnOutput.shape}")


finalOutput = inputToFfnBlock + ffnOutput
print("\n Shape after FFN residual connection: ")
print(f" final output: {finalOutput.shape}")


class SimplifiedLlama4FFN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.hidden_act = config['hidden_act']
        self.ffn_bias = config['ffn_bias']
        self.rms_norm_eps = config['rms_norm_eps']
        self.norm = SimplifiedRMSNorm(self.hidden_size,eps = self.rms_norm_eps)

        self.gate_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=self.ffn_bias)
        self.up_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=self.ffn_bias)
        self.down_proj = nn.Linear(self.intermediate_size,self.hidden_size,bias = self.ffn_bias)

        #Activation

        if self.hidden_act == "silu":
            self.activation_fn = nn.SiLU()

        else:
            raise NotImplementedError(f"Activation {self.hidden_act} not implemented yet.")


    def forward(self,hidden_states):
        normalizedStates = self.norm(hidden_states)

        gate = self.gate_proj(normalizedStates)
        up = self.up_proj(normalizedStates)
        down = self.down_proj(self.activation_fn(gate) * up)

        return down


ffn_config_dict = {
    'hidden_size': hiddenSize,
    'intermediate_size': intermediateSize,
    'hidden_act': hiddenAct,
    'ffn_bias': ffnBias,
    'rms_norm_eps': rmsNormEps,
}

simplifiedFFNModule = SimplifiedLlama4FFN(ffn_config_dict)
mlpOutputFromModule = simplifiedFFNModule(inputToFfnBlock)
finalOutputFromModule = inputToFfnBlock + mlpOutputFromModule


print("\nOutput shape from simplified FFN module (before residual):", mlpOutputFromModule.shape)
print("Output shape after external residual connection:", finalOutputFromModule.shape)
print("Outputs are close:", torch.allclose(finalOutput, finalOutputFromModule, atol=1e-6))




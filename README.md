#  Online RLHF Workbench 🏥

A functional implementation of Online Reinforcement Learning from Human Feedback (RLHF) for medical AI, featuring Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) algorithms.

##  Overview

This project implements a "Control Room" for medical AI that enables:
- Generation of multiple medical reasoning traces with guaranteed diagnostic accuracy
- Human expert ranking of reasoning quality
- Mathematical optimization metrics calculation for model steering
- Healthcare-safe deployment through rejection sampling

Demonstrating post-training optimization techniques used in modern LLMs like GPT-4 and DeepSeek-V3.

## Requirements
torch>=2.0.0
transformers>=4.35.0
gradio>=4.0.0
numpy>=1.24.0
accelerate>=0.24.0
sentencepiece>=0.1.99

### Key Components

#### 1. **Rejection Sampler**
Ensures 100% diagnostic accuracy by filtering generated traces:
- Generates multiple reasoning paths
- Verifies each reaches the correct medical diagnosis
- Maintains healthcare safety standards (zero false negatives)

#### 2. **DPO (Direct Preference Optimization)**
Implements preference learning without explicit reward modeling:
- Calculates implicit rewards: r = log(π_θ/π_ref)
- Computes preference loss: -log(σ(β(r_best - r_worst)))
- Uses reference model to prevent reward hacking

#### 3. **GRPO (Group Relative Policy Optimization)**
DeepSeek-V3 style optimization with variance reduction:
- Assigns group rewards: Best=1.0, Middle=0.5, Worst=0.0
- Normalizes advantages: A = (R - μ)/σ
- Reduces gradient variance for stable training

#### 4. **Model Interface**
Modular design supporting multiple models:
- SmolLM-135M (default, lightweight)
- Phi-1.5 (better quality)
- TinyLlama-1.1B (balanced)
- Easy configuration switching via ModelConfig

1. **Input**: Medical case with question and correct answer
2. **Generation**: Create multiple reasoning traces
3. **Verification**: Ensure all traces reach correct diagnosis
4. **Ranking**: Medical expert evaluates reasoning quality
5. **Optimization**: Calculate DPO and GRPO metrics
6. **Output**: Model update parameters


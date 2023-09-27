# QA-LoRA

This repository provides the PyTorch implementation of [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2309.14717.pdf)

## Installation
```bash
conda create -n qalora python=3.8
conda activate qalora
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
git clone -b peft_integration https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .[triton]
cd ..
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
# CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
# make argument in {cuda110, cuda11x, cuda12x}
# if you do not know what CUDA you have, try looking at the output of: python -m bitsandbytes
CUDA_VERSION=117 make cuda11x
python setup.py install
cd ..
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git
pip install -r requirements.txt
pip install protobuf==3.20.*
```
change the peft_utils.py in your auto-gptq path(python path/auto_gptq/utils/peft_utils.py) with the new one.

## Training
```bash
python qalora.py --quantized_model_path <path>
```

The file structure of the model checkpoint is as follows:
```
config.json             llama7b-4bit-32g.bin  special_tokens_map.json  tokenizer_config.json
generation_config.json  quantize_config.json      tokenizer.model
```

## Quantization
We use [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa) for quantization. 
bits=4, group-size=32, act-order=False
## Acknoledgements
Our code is based on [QLoRA](https://github.com/artidoro/qlora), [GPTQLORA](https://github.com/qwopqwop200/gptqlora), [Auto-GPTQ](https://github.com/PanQiWei/AutoGPTQ/tree/main)

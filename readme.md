# Automated image generation with flusk and phi3 models

## requirements
1. Python 3.10
2. CUDA toolkit 118
3. minGw to build gguf model

## installation
1. Download model [Phi-3-mini-4k-instruct-fp16.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-fp16.gguf)
2. Put model into imageGenerator/promptGenerator/MODELS dir
3. make python venv in root dir
4. in venv run pip install -r requirements.txt
5. copy auto.py to imageGenerator/flux dir
 

## promptGenerator Run
1. activate venv
2. run .\promptGenerator\server\main.py

## imageGenerator Run
1. activate venv
2. run python flux/auto.py --prompts prompts.json --output generated_images --device cuda
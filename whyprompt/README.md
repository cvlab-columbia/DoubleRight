# Environment Setup:

install virtual environment:
`pip install virtualenv`

`virtualenv DROR`

`source DROR/venv/bin/activate`

`pip install -r requirements.txt`

Then replace the files in the replace folder to the source code in your environmet:  

replace `DROR/venv/lib/python3.8/site-packages/torchvision/datasets/files` with files in the replace folder respectively. 


# Evaluation for DoubleRight benchmark on CLIP

Run `CUDA_VISIBLE_DEVICES=4 python eval_clip_fulldoublyright.py`


# Code for Why Pormpt

1. Train ImageNet prompt:
deepprompt: `why_deepprompt_clip_mix.py`

Example: ``CUDA_VISIBLE_DEVICES=0 python why_deepprompt_clip_mix.py --add_prompt_size 30 --batch_size 256 --model_dir=. --deep_prompt --learning_rate 10 --optim=sgd``

Evaluation:

`CUDA_VISIBLE_DEVICES=0 python why_deepprompt_clip_mix.py --add_prompt_size 30 --batch_size 256 --resume=/saved_path/DeepPrompt30_lr10_b256/checkpoint.pth.tar --deep_prompt --evaluate`


2. Train other dataset: `why_prompt_v2_clip.py`

For example
`CUDA_VISIBLE_DEVICES=0 python why_prompt_v2_clip.py --dataset cifar100 --add_prompt_size 3 --batch_size 32`

dataset can be cifar10, food101, caltech101, SUN

Model run evaluation every 5 epoch.



# Install 

```
cd salad
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge fresnel
pip install -e .
```

# Get started

```
python train.py model={phase1, phase2, lang_phase1, lang_phase2}
```

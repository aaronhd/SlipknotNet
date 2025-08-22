# README

# SlipknotNet

This is the code of slipknot detection and emergency stop used for Da Vinci Robotic Surgical Systems, as one robotic application of a principle accepted paper in Nature ("**Slipknot gauged mechanical transmission and intelligent operation**")

Project Website: [https://sites.google.com/view/slipknotnet](https://sites.google.com/view/slipknotnet)

```bash
git clone https://github.com/aaronhd/SlipknotNet.git

cd SlipknotNet

export SlipknotNet_FOLDER=$HOME/code/PointNetGPD

mkdir checkpoint data logging results
```

# 1. Create a virtual environment:

Tested on CUDA 11.8

```bash
conda create -n slipknotnet python=3.10
pip install numpy==1.23.5
pip install opencv-python==4.11.0.86
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install tensorboard==2.18.0
pip install matplotlib==3.10

conda activate slipknotnet
```

# 2. Download our [Davinci-sliputure Dataset](https://www.dropbox.com/scl/fi/czec2557tokjj6h86g9qe/Davinci-sliputure_Dataset.zip?rlkey=pzshhomii64px74ldr8upi4kx&st=lmqmhulv&dl=0)

```bash
# Unzip and move the downloaded datatset into ./data
```

# 3. Train SlipknotNet  model：

```bash
python train.py --network slipknotnet --dataset suture_line --dataset-path ./data/Davinci-sliputure_Dataset  --gpu 0

# The training log is saved in ./logging
```

# 4. Test  model and Inference:

Our [pre-trained model](https://www.dropbox.com/scl/fi/ugwynsjjwblldihu0q04d/checkpoint.zip?rlkey=vqtpal6l2wl4kvsurrq5tywdl&st=8kshisul&dl=0) for testing is available.

```bash
# Unzip and move the downloaded checkpoint into ./checkpoint

python inference.py --network slipknotnet

# The output is saved in ./results

# Test on robotic system
python davinci_slipknotnet_exp.py 
```

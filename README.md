# SlipknotNet README

# SlipknotNet

This is the code of slipknot detection and emergency stop used for Da Vinci Robotic Surgical Systems, as one robotic application of a principle accepted paper in Nature ("**Slipknot gauged mechanical transmission and intelligent operation**")

Project Website: [https://sites.google.com/view/slipknotnet](https://sites.google.com/view/slipknotnet)

```bash
git clone https://github.com/aaronhd/SlipknotNet.git

cd SlipknotNet

mkdir checkpoint data logging results
```

# create a virtual environment

```bash
conda create -n slipknotnet python=3.10
pip install numpy==1.23.5
pip install opencv-python==4.11.0.86
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

conda activate slipknotnet
```

# Download our [Davinci-sliputure Dataset](https://www.dropbox.com/scl/fi/czec2557tokjj6h86g9qe/Davinci-sliputure_Dataset.zip?rlkey=pzshhomii64px74ldr8upi4kx&st=lmqmhulv&dl=0)

```bash
# unzip and move the downloaded datatset into ./data
```

# Train SlipknotNet  model：

```bash
python train.py --network slipknotnet --dataset suture_line --dataset-path ./data/Davinci-sliputure_Dataset  --gpu 0

# The training log is saved in ./logging
```

# Test  model and Inference:

Our [pre-trained model](https://www.dropbox.com/scl/fi/ugwynsjjwblldihu0q04d/checkpoint.zip?rlkey=vqtpal6l2wl4kvsurrq5tywdl&st=8kshisul&dl=0) for testing is available.

```bash
# unzip and move the downloaded checkpoint into ./checkpoint

python inference.py --network slipknotnet

# The output is saved in ./results
```

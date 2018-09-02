## Pervasive Attention: 2D Convolutional Networks for Sequence-to-Sequence Prediction

This is a pytorch implementation of the pervasive attention model:  [Arxiv](https://arxiv.org/abs/1808.03867)


### Requirements (besides pytorch)
```
pip install tensorboardX h5py 
```

### Usage:

#### IWSLT'14 pre-processing:
```
cd data
./prepare-iwslt14.sh
cd ..
python preprocess.py -d iwslt14
```

#### Training:
```
mkdir -p save events
python train.py -c config/l24.yaml
```

#### Generation & evaluation
```
python generate.py -c config/l24.yaml -b 5 
```

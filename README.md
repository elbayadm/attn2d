This is a fork of Fairseq(-py) with implementations of the following models:

## Pervasive Attention - 2D Convolutional Neural Networks for Sequence-to-Sequence Prediction 

An NMT models with two-dimensional convolutions to jointly encode the source and the target sequences.

Pervasive Attention also provides an extensive decoding grid that we leverage to efficiently train wait-k models.

See [README](examples/pervasive/README.md).

## Efficient Wait-k Models for Simultaneous Machine Translation 

Transformer Wait-k models (Ma et al., 2019) with unidirectional encoders and with joint training of multiple wait-k paths.

See [README](examples/waitk/README.md).


# Fairseq Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

To install fairseq:
```bash
pip install fairseq
```

**Installing from source**

To install fairseq from source and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .
```

# License
fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

For Pervasive Attention, please cite:

```bibtex
@InProceedings{elbayad18conll,
    author ="Elbayad, Maha and Besacier, Laurent and Verbeek, Jakob",
    title = "Pervasive Attention: 2D Convolutional Neural Networks for Sequence-to-Sequence Prediction",
    booktitle = "Proceedings of the 22nd Conference on Computational Natural Language Learning",
    year = "2018",
 }
```

For our wait-k models, please cite:

```bibtex
@article{elbayad20waitk,
    title={Efficient Wait-k Models for Simultaneous Machine Translation},
    author={Elbayad, Maha and Besacier, Laurent and Verbeek, Jakob},
    journal={arXiv preprint arXiv:2005.08595},
    year={2020}
}
```

For Fairseq, please cite:

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```


# Transformer Predictor for N-Dimensional Tokens

This code, forked from [Halton-MaskGIT](https://github.com/valeoai/Halton-MaskGIT), is a transformer-based generative predictor based on MaskGIT [(Chang H. et al.)](https://arxiv.org/abs/2202.04200), though this model is intended to be generalised beyond two dimensions.

# How to Use

The main entry point for this project is `main.py`. After setting up an [environment](./environment.yaml), simply running `main.py` will train a model to generate a new examples of data provided by the user.

## Configuring Training

Configuration of training is handled via the [arguments/default_args.yaml](./arguments/default_args.yaml) file, as well as via command line instruction. When `main.py` is run, training follows the configuration in [arguments/default_args.yaml](./arguments/default_args.yaml).

When training begins, a copy of the config file used to begin training is saved by default within the specified logging folder. This can be useful if you want to re-train the model with an already-used config file since you can just run `main.py --config /path/to/log/config.yaml`

### Specifying Custom Config Files

To specify a custom configuration file, run `main.py --config /path/to/config.yaml`.

### Specifying Configurations From the Command Line

From the command line, run `main.py --parameter value` for any configuration parameter and new value required from the standard arguments file. For example, to change the name of the logging folder, run `main.py --name new_name`.

### Configuration Parameters

There are several parameters configurable by the user. The purpose of each is specified as comments within the [arguments/default_args.yaml](./arguments/default_args.yaml).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details. Note this license is inherited from [Halton-MaskGIT](https://github.com/valeoai/Halton-MaskGIT).

## Acknowledgement

### Acknowledgements from Halton-MaskGIT

This project is powered by IT4I Karolina Cluster located in the Czech Republic.

The pretrained VQGAN ImageNet (f=16), 1024 codebook. The implementation and the pre-trained model is coming from the [VQGAN official repository](https://github.com/CompVis/taming-transformers/tree/master)

## BibTeX

If you find our work beneficial for your research, please consider citing both our work and the original source.

```
@misc{besnier2023MaskGit_pytorch,
      title={A Pytorch Reproduction of Masked Generative Image Transformer},
      author={Victor Besnier and Mickael Chen},
      year={2023},
      eprint={2310.14400},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@InProceedings{chang2022maskgit,
  title = {MaskGIT: Masked Generative Image Transformer},
  author={Huiwen Chang and Han Zhang and Lu Jiang and Ce Liu and William T. Freeman},
  booktitle = {CVPR},
  month = {June},
  year = {2022}
}
```

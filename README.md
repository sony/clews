# Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching

_This is the repository for the CLEWS paper. It includes the code to train and evaluate the main models we consider (including baselines), checkpoints for DVI and SHS data sets, and a basic inference script. We do not include the ablation experiments._

### Abstract

Detecting musical versions (different renditions of the same piece) is a challenging task with important applications. Because of the ground truth nature, existing approaches match musical versions at the track level (e.g., whole song). However, most applications require to match them at the segment level (e.g., 20s chunks). In addition, existing approaches resort to classification and triplet losses, disregarding more recent losses that could bring meaningful improvements. In this paper, we propose a method to learn from weakly annotated segments, together with a contrastive loss variant that outperforms well-studied alternatives. The former is based on pairwise segment distance reductions, while the latter modifies an existing loss following decoupling, hyper-parameter, and geometric considerations. With these two elements, we do not only achieve state-of-the-art results in the standard track-level evaluation, but we also obtain a breakthrough performance in a segment-level evaluation. We believe that, due to the generality of the challenges addressed here, the proposed methods may find utility in domains beyond audio or musical version matching.

### Authors

Joan Serrà, R. Oguz Araz, Dmitry Bogdanov, & Yuki Mitsufuji.

### Reference and links

J. Serrà, R. O. Araz, D. Bogdanov, & Y. Mitsufuji (2025). *Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching*. Int. Conf. on Machine Learning (ICML), pp. 53923-53939.

[[`arxiv`](https://arxiv.org/abs/2502.16936)] [[`pmlr`](https://proceedings.mlr.press/v267/serra25a.html)] [[`checkpoints`](https://zenodo.org/records/15045900)]

## Preparation

### Environment

CLEWS requires python>=3.10. We used python 3.10.13.

You should be able to create the environment by running [install_requirements.sh](install_requirements.sh). However, we recommend to just check inside that file and do it step by step.

## Operation

### Inference

We provide a basic inference script to extract embeddings using a pre-trained checkpoint:

```bash
OMP_NUM_THREADS=1 python inference.py --checkpoint=logs/model/checkpoint_best.ckpt --path_in=data/audio_files/ --path_out=cache/extracted_embeddings/
```

It will go through all audio files in the folder and subfolders (recursive) and create the same structure in the output folder. Alternatively, you can use the following arguments for processing just a single file:

```bash
OMP_NUM_THREADS=1 python inference.py --checkpoint=logs/model/checkpoint_best.ckpt --fn_in=data/audio_files/filename.mp3 --fn_out=cache/extracted_embeddings/filename.pt
```

## Training and testing

Note: Training and testing assume you have at least one GPU.

### Folder structure

Apart from the structure of this repo, we used the following folders:
* `data`: folder pointing to original audio and metadata files (can be a symbolic link).
* `cache`: folder where to store preprocessed metadata files.
* `logs`: folder where to output checkpoints and tensorboard files.

You should create/organize those folders prior to running any training/testing script. The folders are not necessary for regular operation/inference.

### Preprocessing

To launch the data preprocessing script, you can run, for instance:

```bash
OMP_NUM_THREADS=1 python data_preproc.py --njobs=16 --dataset=SHS100K --path_meta=data/SHS100K/meta/ --path_audio=data/SHS100K/audio/ --ext_in=mp3 --fn_out=cache/metadata-shs.pt
OMP_NUM_THREADS=1 python data_preproc.py --njobs=16 --dataset=DiscogsVI --path_meta=data/DiscogsVI/meta/ --path_audio=data/DiscogsVI/audio/ --ext_in=mp3 --fn_out=cache/metadata-dvi.pt
```

This script takes time as it reads/checks every audio file (so that you do not need to run checks while training or in your dataloader). You just do this once and save the corresponding metadata file. Depending on the path names/organization of your data set it is possible that you have to modify some minor portions of the `data_preproc.py` script.

### Training

Before every training run, you need to clean the logs path and copy the configuration file (with the specific name `configuration.yaml`):
```bash
rm -rf logs/shs-clews/ ; mkdir logs/shs-clews/ ; cp config/shs-clews.yaml logs/shs-clews/configuration.yaml
rm -rf logs/dvi-clews/ ; mkdir logs/dvi-clews/ ; cp config/dvi-clews.yaml logs/dvi-clews/configuration.yaml
```

Next, launch the training script using, for instance:

```bash
OMP_NUM_THREADS=1 python train.py jobname=shs-clews conf=config/shs-clews.yaml fabric.nnodes=1 fabric.ngpus=2
OMP_NUM_THREADS=1 python train.py jobname=dvi-clews conf=config/dvi-clews.yaml fabric.nnodes=1 fabric.ngpus=2
```

### Testing

To launch the testing script, you can run, for instance:

```bash
OMP_NUM_THREADS=1 python test.py jobname=test-script checkpoint=logs/shs-clews/checkpoint_best.ckpt nnodes=1 ngpus=4 redux=bpwr-10
OMP_NUM_THREADS=1 python test.py jobname=test-script checkpoint=logs/dvi-clews/checkpoint_best.ckpt nnodes=1 ngpus=4 redux=bpwr-10 maxlen=300
```

## License

The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).

## Notes

* If using this code, parts of it, or developments from it, please cite the reference above.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.

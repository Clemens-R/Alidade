# Third-party components

This project references public models, pretrained weights, and libraries but
does not claim ownership of them. The notes below were checked on 24 October
2026 and are a convenience summary, not legal advice. Always review the current
terms at the linked source.

## Training-time models

- [ChiPoint-2](https://huggingface.co/chiikabu-labs/chipoint-2): retrieval
  embeddings and projection heads used to create training targets. Its model
  card identifies the release as MIT licensed. This is specifically ChiPoint-2,
  not the older ChiPoint repository.
- [PLONK OSV-5M](https://huggingface.co/nicolas-dufour/PLONK_OSV_5M): sampled
  coordinate targets used during distillation. Its model card identifies the
  release as MIT licensed.
- [OpenCLIP](https://github.com/mlfoundations/open_clip): loads the ChiPoint
  image towers. OpenCLIP source code is MIT licensed; downloaded pretrained
  weights and their source data can have additional terms.

Teacher weights and gallery files are downloaded by the training scripts and
are not stored in this repository. Their licences and model-card conditions
still apply to training and to any redistributed derivative checkpoint.

## Backbone and software

- MobileNetV3-Large ImageNet initialization is loaded through
  [torchvision](https://github.com/pytorch/vision). The torchvision source is
  BSD-3-Clause licensed. Its maintainers explicitly note that pretrained model
  weights may carry terms inherited from their training data and that users are
  responsible for checking the relevant licences.
- PyTorch, torchvision, NumPy, pandas, Pillow, SciPy, scikit-learn,
  Hugging Face Hub, Transformers, OpenCLIP, and diff-plonk retain their own
  licences.

## Dataset

The challenge images, labels, task statement, and holdout predictions are not
included. The `.gitignore` excludes expected dataset directories and locally
generated prediction files.

## Public-release review

- The full Git history was scanned for credentials, machine paths, datasets,
  caches, and unintended large files before publication.
- The released checkpoint can be loaded with PyTorch's restricted
  `weights_only=True` mode.
- The current all-rights-reserved notice does not grant downstream permission to
  redistribute the checkpoint. A future open-source code licence should state
  explicitly whether it excludes model weights and third-party material.
- ImageNet and challenge-data terms should be reviewed again before relicensing
  or redistributing the checkpoint outside this repository.

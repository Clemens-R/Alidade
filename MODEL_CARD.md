# Alidade model card

## Model summary

This repository contains a compact image-geolocation model for street-level
images from twelve European countries. It predicts a latitude and longitude
from one image. The student combines an ImageNet-initialized MobileNetV3-Large,
1,536 country-balanced geocells, twelve country-routed low-rank experts, and a
country-specific coordinate residual. Four quarter-turn views are averaged at
inference.

The checkpoint contains **4,964,188 PyTorch parameters**. Including BatchNorm
buffers and 1,536 three-dimensional geocell centres, its complete inference
state contains **4,993,242 scalar values**.

## Supported geographic domain

Belarus, Finland, France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden,
Turkey, and the United Kingdom. Predictions outside this domain are unsupported.

## Inputs and outputs

- Input: a street-level RGB image; images are resized to 512 x 512 pixels.
- Output: decimal-degree latitude and longitude.
- Inference: the same model processes rotations of 0, 90, 180, and 270 degrees;
  logits and residuals are averaged before coordinate decoding.

## Training

The student was trained on 11,758 labelled challenge images. No additional
images were added. ImageNet initialization and knowledge distillation from
ChiPoint and PLONK were permitted by the challenge rules. The teachers are used
only during training and are not required for inference.

Training uses a 16-epoch feature-distillation warm-up followed by 56 epochs of
hierarchical geocell and robust angular optimization. Teacher country and cell
distributions are mixed per labelled training image according to teacher
quality.

## Validation results

Five-fold cross-validation, averaged over folds:

| Metric | Result |
|---|---:|
| Median haversine error | 58.554 km |
| Mean haversine error | 333.777 km |
| Within 100 km | 57.944% |
| Within 200 km | 66.219% |
| Within 750 km | 86.137% |
| Country accuracy | 82.113% |

The mean is dominated by rare country mistakes. A true-country oracle achieved
approximately 53.3 km median and 150.4 km mean error, indicating that country
calibration is the main remaining weakness.

## Intended use

- Research and education on parameter-efficient image geolocation.
- Reproduction of the accompanying challenge experiments.
- Evaluation on street-level imagery from the supported countries.

## Limitations and risks

- The model is not a global geolocator.
- It can make very large errors when it selects the wrong country.
- Performance may change for images from different cameras, seasons, years, or
  collection processes.
- Predictions must not be treated as reliable evidence of a person's location.
- Geolocation can create privacy and surveillance risks. Use only on imagery you
  are authorized to process.
- Cross-validation results are from the challenge training distribution; they do
  not guarantee equivalent performance on unrelated datasets.

## Release status

This is a public research artifact. The challenge dataset and holdout
predictions are not distributed. The source and checkpoint remain all rights
reserved unless a later release explicitly grants another licence. See
`LICENSE` and `THIRD_PARTY.md` before reuse or redistribution.

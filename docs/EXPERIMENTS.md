# Experiment summary

The project progressed from direct coordinate regression to a country-aware,
parameter-efficient mixture of experts. Most development numbers below are from
fold 1; the final row is the frozen five-fold average.

| Configuration | Parameters | Median km | Mean km | Main lesson |
|---|---:|---:|---:|---|
| MobileNetV3-Small direct coordinates | 1.32M | 1,203.3 | 1,328.1 | Direct regression averaged incompatible modes. |
| Learned retrieval | 4.86M | 631.9 | 936.5 | Retrieval alone was too coarse and unstable. |
| MobileNetV3-Large 3-D regression | 4.52M | 425.7 | 639.2 | Spherical coordinates helped but did not solve multimodality. |
| Soft geocells plus local decoding | 4.62M | 188.8 | 611.6 | Geographic classification was the decisive output representation. |
| Scratch model, one expert | 4.76M | 146.0 | 527.7 | Capacity without pretrained features was insufficient. |
| ImageNet MobileNetV3, one expert | 3.45M | 83.1 | 336.6 | ImageNet initialization produced the largest single gain. |
| ImageNet MobileNetV4, four experts | 3.32M | 69.3 | 413.3 | Better median, substantially worse tail. |
| MobileNetV3, twelve country experts | 4.20M | 60.8 | 329.2 | Supervised country specialization used capacity effectively. |
| Final: 1,536 cells plus adaptive KD | 4.96M | 55.6 | 323.3 | Complementary teachers improved both local and tail behavior. |
| Frozen five-fold average | 4.96M | 58.6 | 333.8 | Performance generalized reasonably across folds. |

## Teacher findings

- ChiPoint retrieval was locally sharp but sometimes produced catastrophic
  distant matches.
- PLONK was less sharp on average but corrected a useful subset of ChiPoint
  failures.
- GeoCLIP, DINOv3 retrieval, and a Europe-specific StreetCLIP checkpoint were too
  weak to distil globally.
- Four-rotation ChiPoint fusion damaged orientation-sensitive embeddings.
- Adding teachers indiscriminately transferred their biases; adaptive mixing of
  the two complementary teachers was more effective.

## Architecture findings

- More experts helped only after ImageNet initialization stabilized the shared
  representation.
- Country supervision produced more coherent expert domains than scene-cluster
  or broad regional routing.
- Twelve rank-24 experts with top-3 routing fit below the five-million-parameter
  limit because they are low-rank adapters rather than full networks.
- Increasing cell resolution from 1,152 to 1,536 helped more than increasing
  adapter rank without distillation.

## Loss findings

The final loss retained conditional soft-cell cross-entropy, country and router
cross-entropy, robust angular coordinate loss, load balancing, and temperature-2
knowledge distillation. Stronger factorized country losses, hand-designed metric
surrogates, and additional threshold losses did not improve the median and tail
together.

Fold-level results and the selected experiment table are available in
`results/`; the experimental progression graphic is shown in the README.

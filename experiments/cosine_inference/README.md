# Cosine Inference Experiment

**Date:** 2026-03-05  
**Model:** Euclidean-trained (10-way, 5-shot, 15K episodes)  
**Approach:** Override distance to cosine at inference time  

## What was changed
- `predictor_cosine.py`: Cosine distance + L2 normalization + temperature scaling + N-way re-ranking at inference
- `generate_proto_cosine.py`: L2-normalized prototypes, 200 samples per class
- `config_cosine.yaml`: Added temperature=5.0, n_way_inference=10

## Results (5-shot, 37-way: 29 base + 8 custom)
- Base class accuracy: 87.5%
- Few-shot accuracy: 56.2%
- Confidence scores: 4-12% (too flat due to cosine on euclidean-trained model)

## Conclusion
Marginal accuracy gain but confidence scores are poorly calibrated.
The encoder was trained with euclidean distance so magnitude info is wasted by cosine.
Better approach: retrain with cosine end-to-end, or increase n_way in training.

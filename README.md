# AI Water Usage Estimator v3.0

Live demo: https://ai-water-usage-estimator.streamlit.app

Built out of curiosity to understand about the hidden environmental 
costs of AI compute.

## Key Finding
The same GPT-4o request uses 4x more water in Arizona 
than Oregon due to differences in datacenter cooling 
efficiency (WUE). Source: Li et al. 2023.
Prediction: 1000 tokens, GPT-4o, Arizona = 0.013260 liters vs Oregon = 0.003284 liters

## ML Models Compared
| Model | MAE (liters) | Notes |
|-------|-------------|-------|
| Linear Regression (ordinal encoding) | 0.002531 | Baseline |
| Linear Regression (one-hot encoding) | 0.002401 | 5% improvement |
| Decision Tree | 0.000659 | Best model, 72% improvement |

## Feature Importance
1. Model type (gpt-4o vs smaller models): 40%
2. Token count: 38%
3. Datacenter region: 22%

## How It Works

### Deployed (v2.0)
1. User submits a query via Streamlit interface
2. OpenAI API processes query and returns response
3. App captures real token usage from API response
4. Multi-feature linear regression model (built from scratch in NumPy)
   predicts water consumption based on tokens and model type
5. Result displayed dynamically per query

### Research Pipeline (v3.0, not yet wired into deployment)
- Synthetic dataset generation using Li et al. 2023 WUE values per region
- Three models compared: linear regression (ordinal), linear regression 
  (one-hot), decision tree
- Decision tree performs best (MAE 0.000659, 72% improvement over baseline)
- Feature importance: model type (40%), tokens (38%), region (22%)
- Next step: wire the decision tree into the deployed app so predictions
  account for region

## ML Implementation
- v1.0: Hardcoded energy lookup table
- v2.0: Linear regression from scratch in NumPy
- v3.0: Decision tree with location feature, 
        72% better than linear regression

## Tech Stack
Python, NumPy, scikit-learn, Streamlit, OpenAI API

## Citation
Li, P. et al. (2023). Making AI Less Thirsty.
arxiv.org/abs/2304.03271

---
*End-to-end ML engineering: research-backed data 
generation, model comparison, and production deployment.*
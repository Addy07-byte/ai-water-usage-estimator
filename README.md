# AI Water Usage Estimator

An AI-powered web app that estimates the water consumption 
behind every AI query using a machine learning model 
trained with gradient descent from scratch.

Built to raise awareness about the hidden environmental 
costs of AI compute.

## Live Demo
https://ai-water-usage-estimator.streamlit.app/

## What's New in v2.0
Replaced hardcoded energy lookup table with a real 
linear regression model trained using gradient descent 
implemented from scratch in NumPy.

Before: every query showed the same water estimate
After:  water scales dynamically with token usage

Model: f(tokens) = w * tokens + b
w = 0.000519 (found automatically by gradient descent)
b = 0.000001 (found automatically by gradient descent)
Training converged in 1000 iterations from cost 
1.29e-02 to 2.77e-10

## How It Works

1. User submits a query via the Streamlit interface
2. OpenAI API processes the query and returns a response
3. App captures real token usage from the API response
4. Trained ML model predicts water consumption:
   f(tokens) = 0.000519 * total_tokens + 0.000001
5. Result displayed dynamically per query

Longer prompts = more tokens = more water usage.
The model learns this relationship from real data.

## ML Implementation
- Linear Regression implemented from scratch
- Gradient Descent in raw NumPy (no sklearn)
- Training data: real token counts from OpenAI API
- Cost function: Mean Squared Error J(w,b)
- w and b initialized at 0, found automatically

## Built With
- Python, NumPy (ML from scratch)
- Streamlit (frontend and deployment)
- OpenAI API (LLM and token data)
- Streamlit Cloud (hosting)

## Version History
- v1.0: Hardcoded energy lookup table by model type
- v2.0: Linear regression trained with gradient descent
- v3.0 (coming): Multiple features (model type, query complexity)
- v4.0 (coming): Neural network for non-linear relationships

## Future Improvements
- Multiple input features for higher accuracy
- Model comparison dashboard
- Leaderboard for public awareness
- Neural network upgrade
- AWS deployment

## Secrets Managed Securely
OpenAI API key managed via Streamlit Secrets.

---
*This project demonstrates end-to-end ML engineering:
data collection, model training from scratch, and 
production deployment.*

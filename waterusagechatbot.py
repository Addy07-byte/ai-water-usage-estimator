import streamlit as st
import openai
import numpy as np

# 🔐 Secure API setup
client = openai.OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    organization=st.secrets.get("OPENAI_ORG_ID", None)
)

# ============================================
# SPRINT 2 ML MODEL
# Multi-feature linear regression
# Feature 1: total_tokens
# Feature 2: model_type (0=gpt-3.5-turbo, 1=gpt-4o)
# w and b found by gradient descent from scratch
# zscore normalization applied before prediction
# Model-specific water rates:
# gpt-3.5-turbo: 0.000284L per token (lighter model)
# gpt-4o:        0.000519L per token (heavier model)
# ============================================
W_MULTI = np.array([0.11814291, 0.00188326])
B_MULTI = 0.109734
X_MU    = np.array([214.0, 0.6667])
X_SIGMA = np.array([227.64, 0.4714])

def estimate_water(total_tokens, model):
    """
    Predicts water usage using multi-feature linear regression.
    Features: total_tokens + model_type
    w and b found automatically by gradient descent.
    """
    model_type = 0 if model == "gpt-3.5-turbo" else 1
    x_new = np.array([total_tokens, model_type])
    x_norm = (x_new - X_MU) / X_SIGMA
    water = np.dot(x_norm, W_MULTI) + B_MULTI
    return round(max(water, 0), 6)

# 🧠 Page setup
st.set_page_config(page_title="AI Water Usage Estimator", page_icon="💧")
st.markdown("## AI Water Usage Estimator")
st.caption("*Uncover the hidden water cost behind every AI query.*")

# 🎨 Style the Ask AI button
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #0077b6;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# 📝 User Input
prompt = st.text_area("Enter your question:")
model = st.selectbox("Select Model:", ["gpt-3.5-turbo", "gpt-4o"])

# 🚀 Generate answer
if st.button("Ask AI"):
    if prompt:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

        # Capture token data
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Sprint 2 multi-feature prediction
        water_used = estimate_water(total_tokens, model)

        # AI Response
        st.markdown("---")
        st.subheader("🧠 AI Response")
        st.write(answer)

        # Water Estimate
        st.subheader("💧 Estimated Water Used")
        st.metric(label="Liters", value=f"{water_used} L")
        st.caption(f"≈ {round(water_used * 33.8, 1)} fluid ounces (~oz)")
        st.caption("💡 Predicted using multi-feature ML model trained with gradient descent. Features: token count + model type.")

        # Token Usage
        st.markdown("---")
        st.subheader("🔢 Token Usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt Tokens", prompt_tokens)
        col2.metric("Completion Tokens", completion_tokens)
        col3.metric("Total Tokens", total_tokens)

    else:
        st.warning("Please enter a question.")
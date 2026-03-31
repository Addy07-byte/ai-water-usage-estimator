import streamlit as st
import openai

# 🔐 Secure API setup
client = openai.OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    organization=st.secrets.get("OPENAI_ORG_ID", None)
)

# ============================================
# ML MODEL — trained with gradient descent
# w and b found automatically from real token data
# f(tokens) = w * tokens + b
# ============================================
W_TRAINED = 0.000519
B_TRAINED = 0.000001

def estimate_water(total_tokens):
    """
    Predicts water usage using trained linear regression model.
    f(tokens) = w * tokens + b
    w and b found by gradient descent on real API token data.
    """
    water = W_TRAINED * total_tokens + B_TRAINED
    return round(water, 6)


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

        # ADD THIS — capture token data
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        water_used = estimate_water(total_tokens)

        # AI Response
        st.markdown("---")
        st.subheader("🧠 AI Response")
        st.write(answer)

        # Water Estimate
        st.subheader("💧 Estimated Water Used")
        st.metric(label="Liters", value=f"{water_used} L")
        st.caption(f"≈ {round(water_used * 33.8, 1)} fluid ounces (~oz)")

        # ADD THIS — show token data
        st.markdown("---")
        st.subheader("🔢 Token Usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt Tokens", prompt_tokens)
        col2.metric("Completion Tokens", completion_tokens)
        col3.metric("Total Tokens", total_tokens)

    else:
        st.warning("Please enter a question.")




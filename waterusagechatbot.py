import streamlit as st
import openai

# 🔐 Secure API setup
client = openai.OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    organization=st.secrets.get("OPENAI_ORG_ID", None)
)

# ⚡ Constants
MODEL_ENERGY_KWH = {
    "gpt-3.5-turbo": 0.02,
    "gpt-4o": 0.04
}
WATER_PER_KWH = 1.8  # liters per kWh

# 💧 Estimate water used
def estimate_water(model):
    kwh = MODEL_ENERGY_KWH.get(model, 0.02)
    return round(kwh * WATER_PER_KWH, 3)

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
        water_used = estimate_water(model)

        # 🧠 AI Response
        st.markdown("---")
        st.subheader("🧠 AI Response")
        st.write(answer)

        # 💧 Water Estimate
        st.subheader("💧 Estimated Water Used")
        st.metric(label="Liters", value=f"{water_used} L")
        st.caption(f"≈ {round(water_used * 33.8, 1)} fluid ounces (~oz)")

        # 📊 Compare Models
        with st.expander("💡 Compare Water Use Across Models"):
            for m in MODEL_ENERGY_KWH:
                st.write(f"**{m}** ➤ {estimate_water(m)} L")

    else:
        st.warning("Please enter a question.")



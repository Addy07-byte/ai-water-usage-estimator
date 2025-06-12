import streamlit as st
import openai

# 🔐 Paste your OpenAI API Key here
client = openai.OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    organization=st.secrets.get("OPENAI_ORG_ID", None)
)
# Estimated energy usage per model (in kWh per query)
MODEL_ENERGY_KWH = {
    "gpt-3.5-turbo": 0.02,
    "gpt-4o": 0.04  # Adjusted for latest models
}

WATER_PER_KWH = 1.8  # liters per kWh

# Function to estimate water usage
def estimate_water(model):
    kwh = MODEL_ENERGY_KWH.get(model, 0.02)
    return round(kwh * WATER_PER_KWH, 3)

# Streamlit UI
st.set_page_config(page_title="AI Water Usage Chatbot", page_icon="💧")
st.title("💧 AI Water Usage Chatbot")
st.write("🌍 Discover how much water every AI query consumes.")

prompt = st.text_area("Enter your question:")
model = st.selectbox("Select Model:", ["gpt-3.5-turbo", "gpt-4o"])

if st.button("Ask AI"):
    if prompt:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        water_used = estimate_water(model)

        st.subheader("AI Response:")
        st.write(answer)

        st.subheader("Water Used:")
        st.write(f"💧 {water_used} liters (~{round(water_used * 33.8, 1)} fl oz)")
    else:
        st.warning("Please enter a question.")

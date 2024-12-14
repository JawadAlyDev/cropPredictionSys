import streamlit as st

st.title("Visualizations")
if "nitrogenQty" in st.session_state or "phosphorousQty" in st.session_state or "potassiumQty" in st.session_state or "temperatureQty" in st.session_state or "humidityQty" in st.session_state or "pHQty" in st.session_state or "rainfallQty" in st.session_state:
    st.write("You have entered ",st.session_state["nitrogenQty"],st.session_state["potassiumQty"],st.session_state["temperatureQty"],st.session_state["phosphorousQty"],st.session_state["humidityQty"],st.session_state["pHQty"],st.session_state["rainfallQty"])
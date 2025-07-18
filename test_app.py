import streamlit as st

# Simple test app
st.title("🫀 Test App")
st.write("Hello! If you can see this, Streamlit is working!")

st.subheader("Basic Test")
st.write("This is a test to make sure everything works.")

# Simple button
if st.button("Click me!"):
    st.success("Button works!")

# Simple input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

st.write("If you see this page, Streamlit is working correctly.")
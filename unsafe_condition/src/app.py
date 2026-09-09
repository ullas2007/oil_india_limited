import streamlit as st

st.set_page_config(page_title="Unsafe Conditions Portal", page_icon="🚧", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
role = st.sidebar.radio("Select Your Role:", ["Login as Employee", "Login as HSE Manager"])

if role == "Login as Employee":
    import src.employee_app as employee_app
    employee_app.run()
elif role == "Login as HSE Manager":
    import src.manager_app as manager_app
    manager_app.run()
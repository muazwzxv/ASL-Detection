import streamlit as st
import pandas as pd
import numpy as np

st.title('My First app')

st.write("So this is a test of the library streamlit")
st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


check = st.button("Kau gay")
if check:
    st.write("kau masih gay")
elif not check:
    st.write("kimak")


agree = st.checkbox("i agree")
if agree:
    st.write("Great!")

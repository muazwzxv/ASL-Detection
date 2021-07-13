import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.title('Algorithm for american sign language detection')

ayush_knn_path = './exported_model/Ayush/AyushDataset-KNN'
ayush_SVM_path = './exported_model/Ayush/AyushDataset-SVM'

ayush_knn = joblib.load(ayush_knn_path)
ayush_svm = joblib.load(ayush_SVM_path)

st.write("This is the Ayush datasets knn model")
st.write(ayush_knn)

st.write("This is the Ayush datasets svm model")
st.write(ayush_svm)

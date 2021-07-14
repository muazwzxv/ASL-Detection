import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load all necessary files
ayush_knn_path = './exported_model/Ayush/AyushDataset-KNN'
ayush_SVM_path = './exported_model/Ayush/AyushDataset-SVM'
ayush_test_y_path = './testY/Ayush/ayushTestY'
ayush_test_x_path = './testY/Ayush/ayushTestX'
ayush_labels_path = './testY/Ayush/ayushLabels'

ayush_knn = joblib.load(ayush_knn_path)
ayush_svm = joblib.load(ayush_SVM_path)
ayush_test_y = joblib.load(ayush_test_y_path)
ayush_test_x = joblib.load(ayush_test_x_path)
ayush_labels = joblib.load(ayush_labels_path)

add_selectbox = st.sidebar.selectbox(
    'Choose your dataset',
    ('Ayush', 'Khalid', 'Grassknoted', 'Datamunge')
)

st.title('Algorithm for american sign language detection')

st.write("This is the Ayush datasets knn model")
st.write(ayush_knn)

st.write("This is the Ayush datasets svm model")
st.write(ayush_svm)

st.write("Classification Report for Ayush dataset using KNN")
st.write(classification_report(
    ayush_test_y, ayush_knn.predict(ayush_test_x), target_names=ayush_labels.classes_)
)

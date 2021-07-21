from keras.backend import switch
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

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

st.title('Algorithm for american sign language detection')
option = st.sidebar.selectbox(
    'Choose your dataset',
    ('Ayush', 'Khalid', 'Grassknoted', 'Datamunge')
)


def navigate(option):
    if option == 'Ayush':
        ayush_dashboard()
    elif option == 'Khalid':
        st.write('This is the khalid options')
        khalid_dashboard()
    elif option == 'Grassknoted':
        st.write('This is the Grassknoted options')
    else:
        st.write('This is the Datamunge options')


def ayush_dashboard():
    st.write("This is the Ayush datasets knn model")
    st.write(ayush_knn)

    st.write("This is the Ayush datasets svm model")
    st.write(ayush_svm)

    st.write("Classification Report for Ayush dataset using KNN")
    st.write(classification_report(
        ayush_test_y, ayush_knn.predict(ayush_test_x), target_names=ayush_labels.classes_)
    )


def khalid_dashboard():
    accuracy = Image.open('./ss/Khalid/knn/accuracy.JPG')
    classify_report = Image.open('./ss/Khalid/knn/classification report.JPG')
    reevaluate_model = Image.open('./ss/Khalid/knn/reevaluate saved model.JPG')
    st.image(accuracy, caption="Accuracy Khalid dataset", width=700)
    st.image(classify_report, caption="Accuracy Classification report", width=700)
    st.image(reevaluate_model, caption="Accuracy Classification report", width=700)


navigate(option=option)

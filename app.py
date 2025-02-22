import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt=='Linux': pathlib.WindowsPath = pathlib.PosixPath 


st.title('Transport classification model')
file = st.file_uploader('Upload image')
if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('transport_model.pkl')

    pred, pred_id, prob = model.predict(img)

    def translate(text):
        if pred=='Car':
            return 'Avtomobil'
        elif pred=='Boat':
            return 'Suv transporti (qayiq, kema, etc.)'
        elif pred=='Airplane':
            return 'Samalyot'

    st.success(f"Rasmda: {translate(pred)}")
    st.info(f"Aniqlik: {prob[pred_id]*100:.1f}%")
    
    fig = px.bar(x=prob*100,y=model.dls.vocab)
    st.plotly_chart(fig)
else:
    st.info("Rasm kititing!")

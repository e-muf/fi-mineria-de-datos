import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ..helpers import get_table_download_link

# pylint: disable=line-too-long
def write():
  import app
  st.title('Clasificación con Regresión Logística')
  st.markdown('''---''')
  
  try:
    data_url = st.text_input("Data URL:", app.DATA_URL, help="Coloque el URL de su archivo CSV")
    df = pd.read_csv(data_url)
    app.set_data_url(data_url)
  except:
    st.error("Asegurate que la URL contenga un CSV correcto")
    df = pd.read_csv(app.DATA_URL)
    st.text(f"Reporte generado a partir del URL: {app.DATA_URL}")

  st.header('1. Elegir variables de clase')
  cols = st.multiselect("Seleccione las columnas a trabajar:", df.columns)
  X = np.array(df[cols])
  st.dataframe(pd.DataFrame(X, columns=cols))
  st.markdown(get_table_download_link(pd.DataFrame(X, columns=cols)), unsafe_allow_html=True)

  st.header('2. Elección de variable predictora')
  v_pred = st.selectbox('Selecciona la variable predictora:', df.columns)
  le = LabelEncoder()
  Y = le.fit_transform(df[v_pred])
  st.dataframe(pd.DataFrame(Y, columns=[v_pred]))

  st.header('3. Aplicación del algoritmo')
  classifier = linear_model.LogisticRegression()
  seed = 1234
  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=seed, shuffle=True)
  
  st.subheader('X de Entrenamiento')
  st.dataframe(pd.DataFrame(X_train, columns=cols))

  st.subheader('Y de Entrenamiento')
  st.dataframe(pd.DataFrame(Y_train, columns=[v_pred]))

  classifier.fit(X_train, Y_train)
  
  probability = classifier.predict_proba(X_train)
  st.subheader('Predicciones probabilísticas')
  st.dataframe(pd.DataFrame(probability))

  predicts = classifier.predict(X_train)
  st.subheader('Predicciones con clasificación final')
  st.dataframe(pd.DataFrame(predicts, columns=[v_pred]))

  st.subheader(f'Exactitud (Accuracy): {classifier.score(X_train, Y_train)}')  

  st.header('4. Validación del modelo')
  new_pred = classifier.predict(X_validation)
  confusion_matrix = pd.crosstab(Y_validation.ravel(), new_pred, rownames=['Real'], colnames=['Clasificación'])
  st.subheader('Matríz de confusión')
  st.dataframe(confusion_matrix)

  st.subheader(f'Exactitud (Accuracy): {classifier.score(X_validation, Y_validation)}')
  st.subheader(f'Intercept: {classifier.intercept_}')
  st.subheader(f'Coeficients: {classifier.coef_}')

  n_data = {col:0 for col in cols}

  st.header('5. Modelo de clasificación')
  for col in n_data.keys():
    n_data[col] = st.number_input(f'{col}', step=0.01)

  df_pred = pd.DataFrame.from_dict(n_data, orient='index').transpose()
  st.dataframe(df_pred)

  st.subheader(f'Predicción: {classifier.predict(df_pred)}')
   
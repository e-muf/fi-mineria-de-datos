import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import awesome_streamlit as ast
import matplotlib.pyplot as plt
import base64

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import BytesIO

from ..helpers import get_table_download_link

# pylint: disable=line-too-long
def write():
  import app
  st.title('Análisis de Componentes Principales')
  st.markdown('''---''')
  
  try:
    data_url = st.text_input("Data URL:", app.DATA_URL, help="Coloque el URL de su archivo CSV")
    df = pd.read_csv(data_url)
    app.set_data_url(data_url)
  except:
    st.error("Asegurate que la URL contenga un CSV correcto")
    df = pd.read_csv(app.DATA_URL)
    st.text(f"Reporte generado a partir del URL: {app.DATA_URL}")

  st.header("Previsualización del Dataframe")
  st.dataframe(df)

  st.header("1. Estandarización de los datos")
  st.text('Para realizar la normalización de los datos se únicamente se deben tomar en cuenta las columnsa con valores numéricos.')
  dataframe = df.select_dtypes(include=np.number)
  normalized_data = normalize(dataframe)
  st.subheader('Datos normalizados')
  st.dataframe(pd.DataFrame(normalized_data, columns=dataframe.columns))

  st.header("2. Matriz de covarianzas o correlaciones")
  n_components = st.slider('Selecciona el número de componentes (0 es equivalente a None):', 0, len(dataframe.columns), 0, key='n_components')
  if n_components == 0: n_components = None
  pca = pca_components(normalized_data, n_components)
  x_comp = pca.transform(normalized_data)

  st.header('3. Decidir el número de components principales')
  st.markdown('''
  - Se calcula el porcentaje de relevancia, es decir, entre el 75% y 90% de la varianza total.
  - Se identifica mediante una gráfica el grupo de componentes con mayor varianza.
  - Se elige las dimensiones cuya varianza sea mayor a 1.
  ''')
  variance = pca.explained_variance_ratio_
  v_components = st.number_input('Selecciona el número de componentes para la varianza:', 0, len(variance), 0)
  st.text(f"Eigenvalues: {variance}")
  st.subheader(f'Varianza acumulada: {sum(variance[0:v_components])}')

  st.subheader('Gráfica de la varianza acumulada')
  fig, ax = plt.subplots()
  ax.plot(np.cumsum(pca.explained_variance_ratio_))
  ax.set_xlabel('Número de componentes')
  ax.set_ylabel('Varianza acumulada')
  ax.grid()
  fig.tight_layout()

  buf = BytesIO()
  fig.savefig(buf, format='png')
  image_width = 700

  st.image(buf, width=image_width)

  st.header('4. Examinar la proporción de relevancias')
  st.write('La importancia de cada variable se refleja en la magnitd de los valores en los componentes (mayor magnitud es sinónimo de mayor importancia).')
  st.write('Se revisan los valores absolutos de los componentes principales seleccionados.')
  st.dataframe(pd.DataFrame(abs(pca.components_), columns=dataframe.columns))

  cols = st.multiselect("Seleccione las columnas a eliminar:", dataframe.columns)
  dataframeX = dataframe.drop(cols, axis=1)
  st.dataframe(dataframeX)
  st.markdown(get_table_download_link(dataframeX), unsafe_allow_html=True)


def normalize(dataframe):
  normalizer = StandardScaler()
  normalizer.fit(dataframe)
  return normalizer.transform(dataframe)

def pca_components(normalized_data, components=None):
  pca = PCA(n_components=components)
  pca.fit(normalized_data)
  st.subheader('Componentes')
  st.write(pca.components_)
  return pca

  
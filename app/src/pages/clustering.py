import streamlit as st
import awesome_streamlit as ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from io import BytesIO
from kneed import KneeLocator

from ..helpers import get_table_download_link
st.set_option('deprecation.showPyplotGlobalUse', False)
# pylint: disable=line-too-long
def write():
  import app
  st.title('Clustering Particional')
  st.markdown('''---''')
  
  try:
    data_url = st.text_input("Data URL:", app.DATA_URL, help="Coloque el URL de su archivo CSV")
    df = pd.read_csv(data_url)
    app.set_data_url(data_url)
  except:
    st.error("Asegurate que la URL contenga un CSV correcto")
    df = pd.read_csv(app.DATA_URL)
    st.text(f"Reporte generado a partir del URL: {app.DATA_URL}")

  st.header('1. Selección de características')
  st.write('Se utiliza una matriz de correlaciones con el propósito de definir un grupo de características significativas.')

  col1, col2, col3 = st.columns(3)
  with col1:
    hue_col = st.selectbox('Selecciona la variable de agrupación:', df.select_dtypes(exclude=np.number).columns)

  with col2:
    x_col = st.selectbox('X', df.select_dtypes(include=np.number).columns)
  
  with col3:
    y_col = st.selectbox('Y', df.select_dtypes(include=np.number).columns)

  fig, ax = plt.subplots()
  ax = sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col)
  ax.set_title('Gráfico de dispersión')
  ax.set_xlabel(x_col)
  ax.set_ylabel(y_col)
  buf = BytesIO()
  fig.savefig(buf, format='png')
  image_width = 900
  st.image(buf, width=image_width)

  st.subheader('Correlaciones')
  st.dataframe(df.corr(method='pearson'))

  st.subheader('Eliminando variables del dataframe')
  cols = st.multiselect("Seleccione las columnas a eliminar:", df.columns)
  df_work = df.select_dtypes(include=np.number).drop(cols, axis=1)
  st.dataframe(df_work)
  st.markdown(get_table_download_link(df_work), unsafe_allow_html=True)

  st.header('2. Algoritmo K-Means')
  matrix_values = np.array(df_work)
  max_value = st.number_input('Número máximo de clusters', min_value=0, value=12)
  kl = num_clusters(matrix_values, max_value)
  st.subheader(f"Número de clusters recomendando: {kl.elbow}")
  plt.style.use('ggplot')
  kl.plot_knee()
  st.pyplot()
  
  st.header('Crear las etiquetas de los elementos')
  m_particional = clustering(matrix_values, kl.elbow)
  df['cluterP'] = m_particional.labels_
  st.dataframe(df)

  fig, ax = plt.subplots()
  ax.scatter(matrix_values[:,0], matrix_values[:,1], c=m_particional.labels_, cmap='rainbow')
  buf = BytesIO()
  fig.savefig(buf, format='png')
  image_width = 900
  st.image(buf, width=image_width)

  st.header('Crentroides')
  centroides = m_particional.cluster_centers_
  st.dataframe(pd.DataFrame(centroides.round(4), columns=df_work.columns))
  
  
def num_clusters(matrix_values, max_value):
  SSE = []
  for i in range(2, max_value):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(matrix_values)
    SSE.append(km.inertia_)

  kl = KneeLocator(range(2,max_value), SSE, curve='convex', direction='decreasing')
  return kl

def clustering(matrix_values, clusters):
  m_particional = KMeans(n_clusters=clusters, random_state=0).fit(matrix_values)
  m_particional.predict(matrix_values)
  return m_particional
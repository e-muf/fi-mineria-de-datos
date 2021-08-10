import streamlit as st
import pandas as pd
import pandas_profiling
import awesome_streamlit as ast
from streamlit_pandas_profiling import st_profile_report

# pylint: disable=line-too-long
def write():
  import app
  st.title('Análisis Exploratorio de Datos')
  st.markdown('''---''')
  
  try:
    data_url = st.text_input("Data URL:", app.DATA_URL, help="Coloque el URL de su archivo CSV")
    df = pd.read_csv(data_url)
    app.set_data_url(data_url)
  except:
    st.error("Asegurate que la URL contenga un CSV correcto")
    df = pd.read_csv(app.DATA_URL)
    st.text(f"Reporte generado a partir del URL: {app.DATA_URL}")

  pr = df.profile_report()

  st_profile_report(pr)
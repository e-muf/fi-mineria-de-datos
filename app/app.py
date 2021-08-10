import streamlit as st

import awesome_streamlit as ast
import src.pages.eda
import src.pages.pca
import src.pages.clustering
import src.pages.rlm

PAGES = {
    "Análsis Exploratorio de Datos": src.pages.eda,
    "Análsis de Componentes Principales": src.pages.pca,
    "Clustering Particional": src.pages.clustering,
    "Clasificación con Regresión Logística": src.pages.rlm,
}

DATA_URL = "https://raw.githubusercontent.com/e-muf/fi.mineria-de-datos/main/data/WDBCOriginal.csv"
def set_data_url(url):
    global DATA_URL
    DATA_URL = url

def main():
    """Main function of the App"""
    st.set_page_config(layout="wide")
    st.sidebar.title("Menú")
    selection = st.sidebar.radio("Ir a", list(PAGES.keys()))

    st.sidebar.title("Míneria de Datos")
    st.sidebar.info(
        """Proyecto Final 2021-2

Universidad Nacional Autónoma de México

Facultad de Ingeniería 
"""
    )
    st.sidebar.title("Alumno")
    st.sidebar.info(
        """
        Emanuel Flores Martínez
"""
    )

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

    

if __name__ == "__main__":
    main()
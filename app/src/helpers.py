import base64

def get_table_download_link(df):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()
  href = f'<a href="data:file/csv;base64,{b64} " download="dataframe.csv">Download CSV file</a>'
  return href
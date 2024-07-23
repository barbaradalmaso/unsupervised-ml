#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:46:44 2024

@author: barbaradalmaso
@title: unsupervided machine learning project
@overview: in this project I'll use DGC dataset with leukemia patients 
RNA-seq and clinical data) to perform unsupervised machine learning analysis 

"""

# --------------------------------------------------------------
# Part 1: Clustering analysis. In this part I'll perform blood sample grouping 
#         based on gene expression. The selected genes chromossomic translocation
#         is associated with promoting uncontrolled cell proliferation in leukocytes
#         The selected genes are: ['FLT3', 'TP53', 'KMT2A', 'IKZF1', 'NOTCH1', 'PAX5', 'IL7R', 'PTEN', 'TCF3']
# --------------------------------------------------------------

#%% Installing and importing packages
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score



#%% Config wd and importing datasets (samples and its ABL1 expression)
import os 
os.getcwd() # getting wd
os.chdir('/Volumes/Extreme SSD/leukemia-ml') # setting new wd

metadata_files = pd.read_json('/Volumes/Extreme SSD/leukemia-ml/metadata-leukemia/metadata.cart.2024-07-15.json') 
file_names = metadata_files['file_name'] # getting file_names list
merged_files = pd.DataFrame()
for i in file_names:
    file_path = os.path.join('/Volumes/Extreme SSD/leukemia-ml/leukemia-rnaseq/', i)
    if os.path.exists(file_path):    
        file = pd.read_csv(file_path, delimiter='\t', skiprows=1)
        # Extract relevant columns only
        genes_of_interest = ['FLT3', 'TP53', 'KMT2A', 'IKZF1', 'NOTCH1', 'PAX5', 'IL7R', 'PTEN', 'TCF3']
        filtered_file = file[['gene_name', 'tpm_unstranded']].loc[file['gene_name'].isin(genes_of_interest)]
        filtered_file.rename(columns={'tpm_unstranded': i}, inplace=True)
        if merged_files.empty:
            merged_files = filtered_file
        else:
            merged_files = pd.merge(merged_files, filtered_file[['gene_name', i]], on='gene_name', how='outer')
        print(f"{i} successful extracted")
    else:
        print(f"{i} not found")

file = np.transpose(merged_files)
file.columns = file.iloc[0] # Rename first column with first row data
file = file.drop(file.index[0]) # Drop-off first row


#%% Extract leukemia metadata with pathological diagnostic to subtypes ('Acute Myeloid Leukemia (AML)', 'Acute Lymphoblastic Leukemia (ALL)', 'Acute Leukemia of Ambigous Lineage (ALAL'))
metadata_slide = pd.read_csv('/Volumes/Extreme SSD/leukemia-ml/metadata-leukemia/biospecimen.cart.2024-07-15/sample.tsv', delimiter='\t') 

# Colleting case_id for each sample
metadata_files = pd.read_json('/Volumes/Extreme SSD/leukemia-ml/metadata-leukemia/metadata.cart.2024-07-15.json') 
metadata_files = metadata_files.explode('associated_entities').reset_index(drop=True)
metadata_files['case_id'] = metadata_files['associated_entities'].apply(lambda x: x['case_id'])
metadata_files = metadata_files[['case_id', 'file_name']]

# Merging metadata_slide and metadata_files using the case_id column
metadata = pd.merge(metadata_files, metadata_slide, on='case_id', how='inner')
metadata = metadata.drop_duplicates(subset='file_name', keep='first')
metadata = metadata[['file_name', 'tissue_type', 'tumor_code']]
metadata = metadata[metadata['tumor_code'] != "'--"].reset_index(drop=True)
metadata = metadata[metadata['tissue_type'] != "Normal"].reset_index(drop=True)


#%% Apply ZScore and clustering using the different methods (******Hierarchical Clustering******)
print(file.dtypes) # Check file datatype
file = file.apply(pd.to_numeric, errors='coerce') # Transforming file in an numeric variable
file.describe()
file = file.apply(zscore, ddof=1)
file = file.loc[file.index.isin(metadata['file_name'])] # Select files based on metadata filtering (with pathological tumor code)

dissimilarity_metrics = ['euclidean', 'cityblock', 'cosine', 'correlation']
linkage_methods = ['single', 'complete', 'average', 'ward']


# DataFrame para armazenar os clusters de cada método
clusters_df = pd.DataFrame(index=file.index)

# Realizar a clusterização hierárquica e plotar dendrogramas
for metric in dissimilarity_metrics:
    for method in linkage_methods:
        plt.figure(figsize=(10, 7))
        plt.title(f'Dendrogram (Metric: {metric}, Method: {method})')
        
        # Calcular a matriz de dissimilaridade entre as amostras
        dissimilarity_matrix = pdist(file, metric=metric)
        
        # Clusterização hierárquica
        Z = linkage(dissimilarity_matrix, method=method)
        
        # Plotar dendrograma sem os nomes das amostras
        dendrogram(Z, labels=file.index, leaf_rotation=90, no_labels=True)
        
        # Atribuir clusters com base no método e criar o DataFrame com os clusters
        clusters = fcluster(Z, t=3, criterion='maxclust')
        clusters_df[f'{metric}_{method}_cluster'] = clusters
        
        plt.xlabel('Amostras')
        plt.ylabel('Distância')
        plt.show()

#%% Apply ZScore and clustering using the different methods (******K-Means******)
# Definir métricas e valores de k
dissimilarity_metrics = ['euclidean', 'cityblock', 'cosine', 'correlation']
k_values = [2, 3, 4, 5]  # Quantidade de clusters desejada

# DataFrame para armazenar os clusters de cada método
k_means_df = pd.DataFrame(index=file.index)

# Realizar a clusterização k-means e plotar os resultados
for metric in dissimilarity_metrics:
    for k in k_values:
        plt.figure(figsize=(10, 7))
        plt.title(f'k-means Clustering (Metric: {metric}, K={k})')
        
        # Aplicar o k-means nas linhas do DataFrame (amostras)
        kmeans = KMeans(n_clusters=k, random_state=0)
        clusters = kmeans.fit_predict(file)  # Agora passamos o DataFrame diretamente
        
        # Adicionar clusters ao DataFrame
        k_means_df[f'{metric}_k{k}_cluster'] = clusters
        
        # Plotar resultados do k-means sem rótulos no eixo x
        plt.scatter(range(len(file)), clusters, c=clusters, cmap='viridis')
        plt.xlabel('Amostras')
        plt.ylabel('Cluster')
        plt.xticks([])
        plt.show()
#%% Check results for better clustering based on pathological characteristics
clusters_df # Hierarchical clustering
k_means_df # K-means clustering

metadata_hirarchical = pd.merge(metadata, clusters_df, left_on='file_name', right_index=True, how='inner')
metadata_kmeans = pd.merge(metadata, k_means_df, left_on='file_name', right_index=True, how='inner')

#%% Plot graphs

# Hierarchical clustering
  # Df for results storage
  results = pd.DataFrame()

  # Iterar sobre as colunas de clusterização (supondo que as colunas de clusterização estão da 4 em diante)
  for col in metadata_hirarchical.columns[4:]:
      # Agrupar por tipo de tumor e cluster e contar as frequências
      counts = metadata_hirarchical.groupby(['tumor_code', col]).size().unstack(fill_value=0)

      # Adicionar a coluna do método ao DataFrame de resultados
      counts['Method'] = col

      # Adicionar os resultados ao DataFrame principal
      results = pd.concat([results, counts])

  # Plotar os resultados
  for method in results['Method'].unique():
      method_data = results[results['Method'] == method].drop(columns='Method')
      method_data.plot(kind='bar', stacked=True, figsize=(12, 8), title=f'Cluster Distribution for {method}')
      plt.xlabel('Tumor Code')
      plt.ylabel('Frequency')
      plt.show()
      
# K-means clustering
  # Df for results storage
  results_kmeans = pd.DataFrame()

  # Iterar sobre as colunas de clusterização (supondo que as colunas de clusterização estão da 4 em diante)
  for col in metadata_kmeans.columns[4:]:
      # Agrupar por tipo de tumor e cluster e contar as frequências
      counts = metadata_kmeans.groupby(['tumor_code', col]).size().unstack(fill_value=0)

      # Adicionar a coluna do método ao DataFrame de resultados
      counts['Method'] = col

      # Adicionar os resultados ao DataFrame principal
      results_kmeans = pd.concat([results_kmeans, counts])

  # Plotar os resultados
  for method in results_kmeans['Method'].unique():
      method_data = results_kmeans[results_kmeans['Method'] == method].drop(columns='Method')
      method_data.plot(kind='bar', stacked=True, figsize=(12, 8), title=f'Cluster Distribution for {method}')
      plt.xlabel('Tumor Code')
      plt.ylabel('Frequency')
      plt.show()

#%%
"""
Finished on Tue Jul 23 10:46:44 2024

In the end, I was able to apply hierarchical clustering and k-means methods using the leukemia data; 
however, the selected genes were not adequate to correctly segregate the samples.

End.

"""
      
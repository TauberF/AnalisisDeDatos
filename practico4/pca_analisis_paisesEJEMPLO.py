#%% Importar librerias
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#%% Leer csv paises
df_paises = pd.read_csv('Country-data.csv', sep = ',')
df_paises

# %% Graficar la matriz de correlacion de df_paises sin columna 'country'
plt.figure(figsize = (10, 6))
sns.heatmap(df_paises.drop(columns = 'country').corr(), annot = True)
plt.title('Matriz de correlacion de df_paises')
plt.show()

# Realizar analisis de componentes principales
#%% Elegir solo las variables numericas
df_paises_num = df_paises.select_dtypes(include = ['float64', 'int64'])
df_paises_num

#%% Normalizar los datos
scaler = StandardScaler()
df_paises_norm = scaler.fit_transform(df_paises_num)

#%% Crear objeto PCA
pca = PCA(n_components = 9)

#%% Ajustar y transformar los datos
pca_paises = pca.fit_transform(df_paises_norm)

#%% Crear dataframe con los datos transformados
df_pca_paises = pd.DataFrame(data = pca_paises, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])

#%% Realizar biplot mostrando los componentes principales y las variables originales
def biplot(etiquetas, columnas, df_pca, pca, componente_x, componente_y, title = 'Biplot', incluir_nombres = False):
    plt.figure(figsize = (10, 6))
    pcx_txt = f'PC{componente_x}'
    pcy_txt = f'PC{componente_y}'
    plt.scatter(df_pca[pcx_txt], df_pca[pcy_txt], alpha = 0.5)
    plt.title(title)
    plt.xlabel(f'{pcx_txt} {round(pca.explained_variance_ratio_[componente_x-1] * 100, 2)}%')
    plt.ylabel(f'{pcy_txt} {round(pca.explained_variance_ratio_[componente_y-1] * 100, 2)}%')
    for i, (pc1, pc2) in enumerate(zip(pca.components_[componente_x-1], pca.components_[componente_y-1])):
        # Aumentar tamaño de flechas
        pc1 *= 6
        pc2 *= 6
        plt.arrow(0, 0, pc1, pc2, head_width = 0.1, head_length = 0.1, linewidth = 2, color = 'red')
        plt.text(pc1, pc2, columnas[i+1], color = 'black', ha = 'right', va = 'bottom')
    # Mostrar el nombre en cada punto
    if incluir_nombres:
        for i, nombre in enumerate(etiquetas):
            plt.text(df_pca[pcx_txt][i], df_pca[pcy_txt][i], nombre, color = 'blue', ha = 'left', va = 'bottom')
    plt.grid()
    plt.show()

#%% Graficar biplot
biplot(df_paises['country'], df_paises.columns, df_pca_paises, pca, componente_x = 1, componente_y = 2, title = 'Biplot de Paises', incluir_nombres = False)

# %% Graficar la varianza explicada y acumulada en un único gráfico formado por dos subgráficos
plt.figure(figsize = (10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker = 'o')
plt.title('Varianza explicada')
plt.xlabel('Componente')
plt.ylabel('Varianza explicada')
plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker = 'o')
plt.title('Varianza acumulada explicada')
plt.xlabel('Componente')
plt.ylabel('Varianza acumulada')
plt.show()

# %% Graficar la matriz de correlacion de df_paises sin columna 'country'
plt.figure(figsize = (10, 6))
sns.heatmap(df_paises.drop(columns = 'country').corr(), annot = True)
plt.title('Matriz de correlacion de df_paises')
plt.show()

# %% Graficar la matriz de correlacion de los componentes principales
plt.figure(figsize = (10, 6))
sns.heatmap(df_pca_paises.corr(), annot = True)
plt.title('Matriz de correlacion de los componentes principales')
plt.show()

#%% Para cada columna imprimir el valor del componente principal correspondiente a PCx
def imprimir_componentes_principales(df, pca, componente_x):
    pcx_txt = f'PC{componente_x}'
    print(f'Componentes principales para {pcx_txt}')
    for i, columna in enumerate(df.columns):
        print(f'{columna}:'.ljust(15), f'{pca.components_[componente_x-1][i]:.2f}')
        
#%%
imprimir_componentes_principales(df_paises_num, pca, 2)
# %%

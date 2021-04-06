import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shapefile as shp
sns.set(style='whitegrid', palette='pastel', color_codes=True)
sns.mpl.rc('figure', figsize=(10,6))


# Carregando o arquivo shp
amazon_path = 'D:/michel/data/amazonia/forest.shp'
deep_geo_path = 'D:/michel/data/amazonia/deepgeo/prodes_shp_crop.shp'
recife_path = 'C:/Users/miche/PycharmProjects/WebScrapping/Imoveis/RM_Recife/Shapes_RM Recife/RM_Recife_UDH.shp'
sf = shp.Reader(recife_path)

# Podemos descobrir quantas regioes (shapes) existem no arquivo:
N = len(sf.shapes()) # Quantidade de regioes
print(N)

# Vamos explorar mais, usando este comando podemos ler a informação de cada shape individualmente
print(sf.records()[0])

# Vamos criar um dataframe com esse arquivo

def read_shapefile(sf):
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns = fields, data=records)
    df = df.assign(coords=shps)
    return df

df=read_shapefile(sf)
print(df.shape)
df.to_csv('shapefiles.csv', index_label=False)
print(df.head())
# Vamos plotar todos os shapes que estão no dataframe

def plot_map(sf, x_lim=None, y_lim=None, figsize=(11, 9)):
    '''
    Plot map with lim coordinates
    '''
    plt.figure(figsize=figsize)
    id = 0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k')

        if (x_lim == None) & (y_lim == None): # verificando se nao espeficicamos limites
            x0 = np.mean(x) # determinando a posição do titulo da regiao
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=10)
        id = id + 1

    if (x_lim != None) & (y_lim != None): # Se formos usarmos o zoom, não tem legenda
        plt.xlim(x_lim)
        plt.ylim(y_lim)
plot_map(sf)
plt.show()


def plot_map2(id, sf, x_lim=None, y_lim=None, figsize=(11, 9)):
    '''
    Plot map with lim coordinates
    '''

    plt.figure(figsize=figsize)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k')

    shape_ex = sf.shape(id)
    x_lon = np.zeros((len(shape_ex.points), 1))
    y_lat = np.zeros((len(shape_ex.points), 1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    plt.plot(x_lon, y_lat, 'r', linewidth=3)

    if (x_lim != None) & (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

plot_map2(200, sf)
plt.show()
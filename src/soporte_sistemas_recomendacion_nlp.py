# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd

# Para visualización de datos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns

def get_index_from_name(name,columna,df):
    """
    Obtiene el índice de un DataFrame basado en un valor en una columna específica.
    
    Args:
        name (str): Valor buscado en la columna.
        columna (str): Nombre de la columna donde buscar el valor.
        df (pd.DataFrame): DataFrame de pandas.
    
    Returns:
        int: Índice del DataFrame correspondiente al valor.
    """

    return df[df[columna]==name].index[0]


def get_name_from_index(index,df,columna):
    """
    Obtiene el valor de una columna específica basado en el índice del DataFrame.
    
    Args:
        index (int): Índice del DataFrame.
        df (pd.DataFrame): DataFrame de pandas.
        columna (str): Nombre de la columna desde la cual se quiere extraer el valor.
    
    Returns:
        str: Valor correspondiente al índice en la columna especificada.
    """

    return df[df.index==index][columna].values[0]


def plot(peli1, peli2, dataframe):
    """
    Genera un gráfico de dispersión que compara dos películas en un espacio de características.

    Parameters:
    ----------
    peli1 : str
        Nombre de la primera película a comparar.
    peli2 : str
        Nombre de la segunda película a comparar.
    dataframe : pd.DataFrame
        Un dataframe transpuesto donde las columnas representan películas y las filas características.

    Returns:
    -------
    None
        Muestra un gráfico de dispersión con anotaciones para cada película.
    """
    x = dataframe.T[peli1]     
    y = dataframe.T[peli2]

    n = list(dataframe.columns)    

    plt.figure(figsize=(10, 5))

    plt.scatter(x, y, s=0)      

    plt.title('Espacio para {} VS. {}'.format(peli1, peli2), fontsize=14)
    plt.xlabel(peli1, fontsize=14)
    plt.ylabel(peli2, fontsize=14)

    for i, e in enumerate(n):
        plt.annotate(e, (x[i], y[i]), fontsize=12)  

    plt.show();


def filter_data(df):
    """
    Filtra un dataframe de ratings basado en la frecuencia mínima de valoraciones por película y por usuario.

    Parameters:
    ----------
    df : pd.DataFrame
        Un dataframe con columnas 'movieId', 'userId' y 'rating'.

    Returns:
    -------
    pd.DataFrame
        Un dataframe filtrado que contiene solo las películas con al menos 300 valoraciones 
        y los usuarios con al menos 1500 valoraciones.
    """
    ## Ratings Per Movie
    ratings_per_movie = df.groupby('movieId')['rating'].count()
    ## Ratings By Each User
    ratings_per_user = df.groupby('userId')['rating'].count()

    ratings_per_movie_df = pd.DataFrame(ratings_per_movie)
    ratings_per_user_df = pd.DataFrame(ratings_per_user)

    filtered_ratings_per_movie_df = ratings_per_movie_df[ratings_per_movie_df.rating >= 300].index.tolist()
    filtered_ratings_per_user_df = ratings_per_user_df[ratings_per_user_df.rating >= 1500].index.tolist()
    
    df = df[df.movieId.isin(filtered_ratings_per_movie_df)]
    df = df[df.userId.isin(filtered_ratings_per_user_df)]
    return df







class AnalisisSentimientos:
    """
    Clase para realizar análisis de sentimientos en un dataframe
    y generar visualizaciones basadas en los resultados.
    """
    def __init__(self, dataframe, columna_texto):
        """
        Inicializa el analizador de sentimientos y prepara el dataframe.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame que contiene los datos.
        columna_texto : str
            Nombre de la columna que contiene los textos a analizar.
        """
        self.dataframe = dataframe.copy()
        self.columna_texto = columna_texto
        self.sia = SentimentIntensityAnalyzer()
        self._preparar_datos()
    
    def _preparar_datos(self):
        """
        Aplica el análisis de sentimientos y separa las puntuaciones en columnas individuales.
        """
        self.dataframe['scores_sentimientos'] = self.dataframe[self.columna_texto].apply(self._analizar_texto)
        self.dataframe['neg'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['neg'])
        self.dataframe['neu'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['neu'])
        self.dataframe['pos'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['pos'])
        self.dataframe['compound'] = self.dataframe['scores_sentimientos'].apply(lambda x: x['compound'])
        self.dataframe.drop(columns=['scores_sentimientos'], inplace=True)
    
    def _analizar_texto(self, texto):
        """
        Analiza un texto y retorna las puntuaciones de sentimientos.

        Parameters:
        ----------
        texto : str
            Texto a analizar.

        Returns:
        -------
        dict
            Diccionario con las puntuaciones de negatividad, neutralidad, positividad y compound.
        """
        return self.sia.polarity_scores(texto)
    
    def graficar_distribucion_sentimientos(self):
        """
        Genera un gráfico de barras para visualizar la distribución de sentimientos (neg, neu, pos).
        """
        mean_scores = self.dataframe[['neg', 'neu', 'pos']].mean()
        mean_scores.plot(kind='bar', figsize=(8, 5), title='Distribución Promedio de Sentimientos')
        plt.xlabel('Tipo de Sentimiento')
        plt.ylabel('Puntuación Promedio')
        plt.xticks(rotation=0)
        plt.show()

    def graficar_distribucion_compound(self):
        """
        Genera un histograma para visualizar la distribución de las puntuaciones compound.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.dataframe['compound'], bins=20, kde=True, color='blue')
        plt.title('Distribución de Puntuaciones Compound')
        plt.xlabel('Puntuación Compound')
        plt.ylabel('Frecuencia')
        plt.show()

    def graficar_mapa_calor_sentimientos(self):
        """
        Genera un mapa de calor para visualizar las correlaciones entre las puntuaciones de sentimientos.
        """
        matriz_corr = self.dataframe[['neg', 'neu', 'pos', 'compound']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Mapa de Calor de Correlaciones de Sentimientos')
        plt.show()

    def obtener_resumen(self):
        """
        Retorna un resumen estadístico de las puntuaciones de sentimientos.

        Returns:
        -------
        pd.DataFrame
            Resumen estadístico (count, mean, std, min, max) de las puntuaciones.
        """
        return self.dataframe[['neg', 'neu', 'pos', 'compound']].describe()
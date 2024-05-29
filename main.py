from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()


# Montar la carpeta de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Función para cargar los DataFrames bajo demanda
def cargar_dataframes():
    df_games = pd.read_parquet(r'Data/parquet/games.parquet')
    df_review = pd.read_parquet(r'Data/parquet/review_predicha.parquet')
    df_item = pd.read_parquet(r'Data/parquet/items.parquet')
    return df_games, df_review, df_item

# Función para inicializar el modelo y el vectorizador TF-IDF
def inicializar_modelo(df_games):
    # Combinamos las características de géneros y etiquetas en una sola columna
    df_games['combined_features'] = df_games['genres'].fillna('') + ' ' + df_games['tags'].fillna('')
    # Inicializamos el vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    # Transformamos las características combinadas en una matriz TF-IDF
    caracteristicas = vectorizer.fit_transform(df_games['combined_features'])
    # Inicializamos el modelo de vecinos más cercanos usando la métrica del coseno y el algoritmo brute force
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    # Entrenamos el modelo con la matriz TF-IDF de características
    model.fit(caracteristicas)
    # Devolvemos el modelo entrenado, el vectorizador y la matriz de características
    return model, vectorizer, caracteristicas

# Cargar los DataFrames y inicializar el modelo al inicio de la aplicación
df_games, df_review, df_item = cargar_dataframes()
model, vectorizer, caracteristicas = inicializar_modelo(df_games)

### Funciones para Alimentar la API
@app.get("/", response_class=HTMLResponse)
async def inicio():
    template = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>API Steam</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                    background-color: black; /* Set background color to black */
                    color: yellow; /* Set text color to yellow */
                }
                h1 {
                    text-align: center;
                }
                p {
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
                img {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 50%; /* Adjust the width as needed */
                }
            </style>
        </head>
        <body>
            <h1>Proyecto individual 1</h1>
            <p>Para acceder a las funciones coloque <code>/docs</code> en la URL</p>
            <img src="/static/logo-henry.png">
        </body>
    </html>
    """
    return HTMLResponse(content=template)

@app.get("/developer/{empresa}")
def developer(empresa: str):
    # Copiamos el DataFrame de juegos
    df_empresa = df_games.copy()
    # Filtramos los juegos desarrollados por la empresa especificada (ignorando mayúsculas)
    df_empresa = df_empresa[df_empresa['developer'].str.lower() == empresa.lower()]
    # Agrupamos por año
    grouped = df_empresa.groupby('anio')
    resultados = []
    # Calculamos el número de ítems y el porcentaje de ítems gratuitos por año
    for year, group in grouped:
        total_items = len(group)
        free_items = len(group[group['price'] == 0])
        porcentaje_free = (free_items / total_items) * 100 if total_items > 0 else 0
        
        # Añadimos los resultados a la lista
        resultados.append({
            'Año': year,
            'Cantidad de Items': total_items,
            'Porcentaje Free': porcentaje_free
        })
    # Devolvemos los resultados
    return resultados

@app.get("/userdata/{user_name}")
def userdata(user_name: str) -> dict:
    # Copiamos los DataFrames de ítems y reviews
    user_items = df_item.copy()
    user_reviews = df_review.copy()
    # Filtramos los ítems del usuario especificado
    user_items = user_items[user_items['user_id'] == user_name]
    # Contamos el número de ítems únicos
    num_items = len(user_items['item_id'].unique())
    # Unimos los ítems del usuario con el DataFrame de juegos para obtener los precios
    user_items_prices = user_items.merge(df_games, left_on='item_id', right_on='id', how='inner')
    # Calculamos el total gastado
    total_gastado = user_items_prices['price'].sum()
    # Filtramos las reviews del usuario especificado
    user_reviews = user_reviews[user_reviews['user_id'] == user_name]
    # Contamos el número total de reviews y el número de reviews positivas
    total_reviews = len(user_reviews)
    positive_reviews = user_reviews[user_reviews['recommend'] == 2]
    num_positive_reviews = len(positive_reviews)
    # Calculamos el porcentaje de reviews positivas
    porcentaje_positive_reviews = (num_positive_reviews / total_reviews) * 100 if total_reviews > 0 else 0
    # Creamos un diccionario con los resultados
    user_data = {
        'Total gastado (USD)': total_gastado,
        'Porcentaje de recomendación positiva': porcentaje_positive_reviews,
        'Cantidad de juegos del usuario': num_items
    }
    # Devolvemos los resultados
    return user_data

@app.get("/best_developer_year/{year}")
def best_developer_year(year: int):
    # Copiamos los DataFrames de juegos y reviews
    games_df = df_games.copy()
    reviews_df = df_review.copy()
    # Ordenamos los DataFrames por 'id' e 'item_id'
    games_df.sort_values(by='id', ascending=False, inplace=True, ignore_index=True)
    reviews_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    # Filtramos los juegos del año especificado
    games_filtered = games_df[games_df['anio'] == year]
    # Unimos los juegos filtrados con las reviews
    merged_df = pd.merge(games_filtered, reviews_df, left_on='id', right_on='item_id')
    # Contamos las recomendaciones positivas por desarrollador
    developer_counts = merged_df[merged_df['recommend'] == 2]['developer'].value_counts().reset_index()
    developer_counts.columns = ['Developer', 'Recommendations']
    # Ordenamos los desarrolladores por número de recomendaciones
    sorted_developers = developer_counts.sort_values(by='Recommendations', ascending=False)
    # Seleccionamos los 3 mejores desarrolladores
    top_developers = sorted_developers.head(3)
    # Creamos una lista con los resultados
    result = [{"Puesto {}: {}".format(i+1, row['Developer']): row['Recommendations']} for i, row in top_developers.iterrows()]
    # Devolvemos los resultados
    return result

@app.get("/developer_reviews_analysis/{desarrolladora}")
def developer_reviews_analysis(desarrolladora: str):
    # Convertimos el nombre del desarrollador a minúsculas
    desarrolladora = desarrolladora.lower()
    # Copiamos los DataFrames de juegos y reviews
    games_df = df_games.copy()
    reviews_df = df_review.copy()
    # Ordenamos los DataFrames por 'id' e 'item_id'
    games_df.sort_values(by='id', ascending=False, inplace=True, ignore_index=True)
    reviews_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    # Unimos los juegos con las reviews
    merged_df = pd.merge(games_df, reviews_df, left_on='id', right_on='item_id')
    # Filtramos los juegos del desarrollador especificado
    developer_filtered = merged_df[merged_df['developer'] == desarrolladora]
    # Contamos las reviews positivas y negativas
    positive_count = len(developer_filtered[developer_filtered['sentiment_analysis'] == 2])
    negative_count = len(developer_filtered[developer_filtered['sentiment_analysis'] == 0])
    # Creamos un diccionario con los resultados
    result = {desarrolladora: {'Negative': negative_count, 'Positive': positive_count}}
    # Devolvemos los resultados
    return result

@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
    # Capitalizamos el nombre del género
    genero = genero.capitalize()
    # Filtramos los juegos del género especificado directamente
    genre_games = df_games[df_games[genero] == 1]
    # Unimos los ítems con los juegos del género
    genre_items = pd.merge(df_item, genre_games, left_on='item_id', right_on='id')
    # Eliminamos los valores nulos en 'playtime_forever' y convertimos a entero
    genre_items = genre_items[~genre_items['playtime_forever'].astype(str).str.contains('None')].copy()
    genre_items['playtime_forever'] = genre_items['playtime_forever'].astype(int)
    # Filtramos los ítems con un tiempo de juego menor o igual a 8760 horas
    filtered_items = genre_items[genre_items['playtime_forever'] <= 8760]
    # Agrupamos por usuario y año, sumando el tiempo de juego
    grouped_items = filtered_items.groupby(['user_id', 'anio'])['playtime_forever'].sum().reset_index()
    # Limitamos el tiempo de juego a un máximo de 8760 horas
    grouped_items['playtime_forever'] = grouped_items['playtime_forever'].clip(upper=8760)
    # Identificamos al usuario con más horas jugadas
    top_user = grouped_items.groupby('user_id')['playtime_forever'].sum().idxmax()
    # Obtenemos las horas jugadas por año del usuario con más horas jugadas
    top_user_hours = grouped_items[grouped_items['user_id'] == top_user]
    acumulacion_horas_anio = top_user_hours.groupby('anio')['playtime_forever'].sum().reset_index()
    # Creamos una lista con las horas jugadas por año
    horas_jugadas_por_anio = [{'anio': anio, 'horas': horas} for anio, horas in zip(acumulacion_horas_anio['anio'], acumulacion_horas_anio['playtime_forever'])]
    # Creamos un diccionario con los resultados
    resultado = {
        f'Usuario con más horas jugadas para género {genero}:': top_user,
        'Horas jugadas': horas_jugadas_por_anio
    }
    # Devolvemos los resultados
    return resultado

@app.get('/recomendacion_juego/{juego_id}')
def recomendacion_juego(juego_id: int):
    # Verificamos si el juego existe en el DataFrame de juegos
    if juego_id not in df_games['id'].unique():
        raise HTTPException(status_code=404, detail="Juego no encontrado")
    # Obtenemos las recomendaciones de juegos similares
    recomendaciones = get_recommendations(juego_id)
    # Filtramos las recomendaciones para eliminar títulos nulos
    recomendaciones_filtradas = [idx for idx in recomendaciones if not pd.isnull(df_games.iloc[idx]['title'])]
    # Obtenemos los títulos de los juegos recomendados
    titulos_recomendados = [df_games.iloc[idx]['title'] for idx in recomendaciones_filtradas]
    # Devolvemos las recomendaciones
    return {
        "juego_id": juego_id,
        "recomendaciones": titulos_recomendados
    }

def get_recommendations(item_id, k=5):
    # Obtenemos el índice del juego en el DataFrame
    juego_idx = df_games[df_games['id'] == item_id].index
    if len(juego_idx) == 0:
        return ["No hay juegos recomendados"]
    else:
        juego_idx = juego_idx[0]
        # Calculamos las distancias y los índices de los juegos más cercanos
        distances, indices = model.kneighbors(caracteristicas[juego_idx], n_neighbors=k+1)
        indices = indices[:, 1:]
        # Devolvemos los índices de los juegos recomendados
        if len(indices) == 0:
            return ["No hay juegos recomendados"]
        else:
            return indices.flatten().tolist()
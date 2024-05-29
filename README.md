# Machine Learning Operations (MLOps)

## Descripción
El objetivo del proyecto es desarrollar un sistema de recomendación de videojuegos para la plataforma Steam. 
Se emplea la Extracción, Transformación y Carga (ETL) de los datos para limpiarlos y normalizarlos.
Luego, se realiza un análisis exploratorio de datos (EDA) para comprender mejor la información disponible y se construye un modelo de machine learning para generar recomendaciones personalizadas basadas en las preferencias de los usuarios.

Se crean diferentes respuestas a consultas, a partir de funciones:
- Cantidad de items y porcentaje de contenido gratuito por año, según la empresa desarrolladora.
- Cantidad de dinero gastado por el usuario, el porcentaje de recomendaciones basado en reseñas y la cantidad de items del mismo.
- Determinar el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
- Devolver el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado.
- Devolver la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.

Por utlimo se implementa el modelo de aprendizaje automático que brinda un sistema de recomendación. El modelo responde a una relación ítem-ítem, es decir, toma un ítem y recomienda otros similares basados en su similitud.

## Dependencias
- python 3.x, en este caso se trabajo con python 3.10
- pip
- pandas
- matplotlib
- seaborn
- uvicorn
- fastapi
- nltk
- fastparquet
- numpy

Esas son las principales, el resto está en el archivo requeriments.txt

### Deploy en Render
Para el deploy de la API se seleccionó la plataforma Render. El servicio esta corriendo en [https://proyecto-individual-1-tbv2.onrender.com](https://proyecto-individual-1-tbv2.onrender.com).

### Video
En el siguiente enlace se encuentra el [video]() con una explicación breve de la API.

---

## Fuente de Datos

**Dataset:** [Carpeta con los archivos que requieren ser procesados](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj). Tengan en cuenta que algunos datos están anidados (un diccionario o una lista como valores en la fila).

**Diccionario de Datos:** [Diccionario con algunas descripciones de las columnas disponibles en el dataset](https://docs.google.com/spreadsheets/d/1-t9HLzLHIGXvliq56UE_gMaWBVTPfrlTf2D9uAtLGrk/edit?usp=drive_link).
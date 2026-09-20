# Utilitzem una imatge oficial de Python lleugera
FROM python:3.10-slim-bullseye
RUN apt-get update && apt-get install -y libexpat1

# Definim el directori de treball a dins del contenidor
WORKDIR /app

##

# Copiem el fitxer de dependències primer (per aprofitar la memòria cau de Docker)
COPY requirements.txt .

# Instal·lem les dependències de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiem la resta del codi del projecte i la carpeta de l'IA a dins del contenidor
COPY . .

# La clau del service account de Google Earth Engine NO es copia a la imatge (.dockerignore).
# Cal muntar-la en executar el contenidor, p. ex.:
#   docker run --gpus all -p 8501:8501 -v /ruta/credentials2.json:/app/credentials2.json:ro <imatge>
# (o bé muntar-la on vulguis i indicar-ne la ruta amb -e GEE_CREDENTIALS=/ruta/dins/contenidor.json)

# Exposem el port que utilitza Streamlit per defecte
EXPOSE 8501

# Comanda per engegar la web automàticament quan arrenqui el contenidor
CMD ["streamlit", "run", "website.py", "--server.address=0.0.0.0"]
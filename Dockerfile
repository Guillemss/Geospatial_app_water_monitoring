# Utilitzem una imatge oficial de Python lleugera
FROM python:3.10-slim

# Definim el directori de treball a dins del contenidor
WORKDIR /app

# Copiem el fitxer de dependències primer (per aprofitar la memòria cau de Docker)
COPY requirements.txt .

# Instal·lem les dependències de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiem la resta del codi del projecte i la carpeta de l'IA a dins del contenidor
COPY . .

# Creem la carpeta oculta de Google a dins del Linux del Docker
RUN mkdir -p /root/.config/earthengine/

# Copiem el teu arxiu de Windows cap a dins del Docker
COPY credentials /root/.config/earthengine/credentials

# Exposem el port que utilitza Streamlit per defecte
EXPOSE 8501

# Comanda per engegar la web automàticament quan arrenqui el contenidor
CMD ["streamlit", "run", "website.py", "--server.address=0.0.0.0"]
import streamlit as st
import os
import matplotlib.pyplot as plt
import folium #per poder posar un mapa
from streamlit_folium import st_folium
from folium.plugins import Draw
import datetime
import data_extraction_2
import shutil #ens permet esborrar arxius

import data_extraction_2
import gpu_processing
import analisis

import importlib
importlib.reload(gpu_processing)
importlib.reload(analisis) #per forçar l'acualització dels arxius
importlib.reload(data_extraction_2)


#-----
# --- CONNEXIÓ A EARTH ENGINE UNA VEGADA INICIALMENT ---
@st.cache_resource
def iniciar_connexio_satelit():
    try:
        data_extraction_2.inicialitzar_gee()
    except Exception as e:
        st.error(f"Error connectant a Earth Engine: {e}")
iniciar_connexio_satelit()


#------------------------------------------------

st.set_page_config(
    page_title = "Water Monitoring",
    page_icon = "💧",
    layout="wide"
    ) #per a que pugui utilitzar tota la pàgina

#---------- CODI CSS PER PODER MODIFICAR COSES QUE STREAMLIT NO ENS PERMET --------------


st.markdown("""
<style>
/* Estil per defecte del botó */
    div.stButton > button:first-child {
        background-color: #28a745; /* Color verd per defecte */
        color: white;
        font-size: 24px; /* <--- MIDA DE LA LLETRA (Pots canviar el 24 pel número que vulguis) */
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #28a745;
        padding: 10px 24px; /* Fa que el botó tingui més espai per dins */
        transition: all 0.3s ease; /* Fa que qualsevol canvi visual sigui fluid i no de cop */
    }
/* Efecte HOVER (Quan el ratolí passa per sobre) */
    div.stButton > button:first-child:hover {
        background-color: #1e7e34; /* Es torna un verd una mica més fosc */
        border: 2px solid #1e7e34;
        color: white;
        transform: scale(1.05); /* <--- EFECTE: Es fa un 5% més gran de mida */
        box-shadow: 0px 8px 15px rgba(40, 167, 69, 0.4); /* <--- EFECTE: Crea una ombra verdosa sota el botó */
    }

/* Efecte ACTIVE (Quan fas el clic just en aquell moment) */
    div.stButton > button:first-child:active {
        transform: scale(0.95); /* Fa l'efecte de prémer el botó cap endins */
    }
    /* --- CANVIAR L'ESTIL DEL MARCADOR --- */
    /* 1. Fer la lletra més gran i en negreta */
    div[data-testid="stCheckbox"] label p {
        font-size: 15px !important; /* Pots fer aquest número més gran o més petit */
        font-weight: bold !important; /* Això posa la lletra en negreta */ste
    }
    
    /* 2. Fer el quadradet de la casella més gran perquè quedi compensat */
    div[data-testid="stCheckbox"] div[role="checkbox"] {
        transform: scale(1); /* Augmenta la mida del quadrat un 50% */
        margin-right: 15px; /* Deixa una mica d'aire entre el quadrat i el text */
        margin-left: 5px;
    }
    /* --- REDUIR L'ESPAI BLANC SUPERIOR --- */
    .block-container {
        padding-top: 1rem !important; /* Redueix el marge de dalt. Per defecte és 6rem */ /* ES POT CANVIAR LA POSICIÓ MODIFICANT 2REM: 1rem,0rem...
        padding-bottom: 1rem !important; /* Redueix el marge de baix per si de cas */
    }

    /* --- CENTRAR LES MÈTRIQUES DE NÚVOLS I AIGUA --- */
    div[data-testid="stMetric"] {
        text-align: center !important;
    }
    div[data-testid="stMetricValue"] > div {
        justify-content: center !important;
    }

    /* --- REDUIR L'ESPAI BLANC SUPERIOR --- */
    .block-container {
        padding-top: 1rem !important; 
        padding-bottom: 1rem !important; 
    }
    #
    
    /* --- CENTRAR LES MÈTRIQUES DE NÚVOLS I AIGUA --- */
    div[data-testid="stMetric"] {
        text-align: center !important;
    }
    div[data-testid="stMetricValue"] > div {
        justify-content: center !important;
    }
</style>

</style>
""", unsafe_allow_html = True)

st.title("Satellite Edge Computing: Water Level Monitoring")
st.write("This application simulates data processing with GPU's on a satellite and shows water evolution over time.")

# 1 Carpeta de dades interna del servidor (L'usuari no la veu)
ruta_carpeta = os.path.join(os.getcwd(), 'dades_satelit_temporals')



#----------------------------------------------------------------------------------------------------------
#AFEGIR UNA SELECCIÓ DE DATES:
st.write("🗓️ Selecciona l'interval de temps:")
col_data1, col_data2 = st.columns(2) #Creem dues columnes

data_minima_s2 = datetime.date(2017,1,1)
data_avui = datetime.date.today()

historic_activat = st.checkbox("📊 Generar gràfic històric complet (2017- Avui)")

#quan posme with li estem dient a python que tot el que posem dins del with volem que ho posi dins de la columna
with col_data1:
    data_inci = st.date_input(
        "Data d'inici", 
        value = datetime.date(2026,1,1),
        min_value = data_minima_s2,
        max_value = data_avui,
        disabled = historic_activat
        )

with col_data2:
    # Data per defecte: Avui
    data_final = st.date_input(
        "Data final", 
        value = datetime.date.today(),
        min_value = data_minima_s2,
        max_value= data_avui,
        disabled = historic_activat
        )

#AFEGIR UN MAPA INTERACTIU:
st.subheader("📍 Selecciona l'àrea d'interès del mapa:")

# Coordenades aproximades del Pantà de Sau
latitud_sau = 41.986
longitud_sau = 2.398

# Creem el mapa centrat a Sau amb Folium
#Mapa per defecte: m = folium.Map(location=[latitud_sau, longitud_sau], zoom_start=12)
# 1. Crear el mapa amb l'eina de dibuix activada (Amb capa de satèl·lit de Google)
m = folium.Map(
    location=[41.986, 2.398], 
    zoom_start=8, # He allunyat una mica el zoom per veure més territori
    tiles="http://mt0.google.com/vt/lyrs=y&hl=ca&x={x}&y={y}&z={z}",
    #posar lyrs=y si volem que surtin noms + satèlit
    #lyrs=s si volem que sigui només imatge de satèl·lit
    attr="Google"
)

#Afegim les eines de dibuix:
draw = Draw(
    export=False,
    draw_options={
        'polyline': False,
        'polygon': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'rectangle': True, # Només permetre rectangles
    }
)
draw.add_to(m)

# Afegim un marcador a la zona
#folium.Marker(
#    [latitud_sau, longitud_sau], 
#    #popup="Pantà de Sau - Zona d'Estudi Sentinel-2",
#    #icon=folium.Icon(color="blue", icon="tint")
#).add_to(m)

# Dibuixem el mapa dins de Streamlit
#st_folium(m, width=700, height=500)
output_mapa = st_folium(m, height=500, use_container_width=True, returned_objects =["all_drawings"])

coordenades_rectangle = None

st.sidebar.markdown("### 🗺️ Instruccions del Mapa:")
st.sidebar.write(
    "Per seleccionar l'àrea d'estudi:\n"
    "1. Busca la icona del **quadrat** a la barra d'eines de l'esquerra del mapa.\n"
    "2. Fes clic i **arrossega** per delimitar la zona.\n"
    "3. El sistema processarà automàticament les dades de la zona triada."
)

#PER POSAR UNA BARRA PEL PERCENATGE DE NÚVOLS QUE VOLEM
#st.sidebar.header("☁️ Filtres d'imatge")
#nivell_nuvols = st.sidebar.slider(
#    "Màxim % de núvols permès", 
#    min_value=0, 
#    max_value=50, 
#    value=10
#)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Sobre el projecte")
st.sidebar.info(
    "Visor geoespacial desenvolupat per monitoritzar l'estat de l'aigua "
    "mitjançant teledetecció i Google Earth Engine."
)


if output_mapa and output_mapa.get("all_drawings"):
    # Obtenim l'últim element dibuixat per l'usuari
    ultim_dibuix = output_mapa["all_drawings"][-1]
    
    if ultim_dibuix["geometry"]["type"] == "Polygon":
        # Extraiem les coordenades de la caixa
        coordenades_rectangle = ultim_dibuix["geometry"]["coordinates"][0]

        st.session_state['coordenades_guardades'] = coordenades_rectangle

        st.success("S'ha seleccionat una zona correctament al mapa!")




#------------------------------------------------------------------------------------------------------
#2. AFEGIM UN BOTÓ PER EXECUTAR EL PROCÉS

#st.write("---")
col1,col2,col3 = st.columns([1,2,1])

with col2:
    boto_executat = st.button("🚀 Executar Processament (GPU)", use_container_width=True)


if boto_executat:
    coordenades_rectangle = st.session_state.get('coordenades_guardades', None)
    #Comprovem si l'usuari ha adibuixat un rectangle
    if coordenades_rectangle is None:
        st.warning("⚠️ Si us plau, dibuixa un rectangle al mapa abans de començar el processament.")

    else:
        #Convertir el polígon de Folium a BBOX (lon_min, lat_min, lon_max, lat_max)
        lons=[punt[0] for punt in coordenades_rectangle]
        lats = [punt[1] for punt in coordenades_rectangle]
        bbox_calculat = [min(lons), min(lats), max(lons), max(lats)]

        if historic_activat:
            data_ini_str = '2017-03-28'
            data_fin_str = data_avui.strftime('%Y-%m-%d')#Posar les dades en format Earth Engine (YYYY-MM-DD)
        else:
            data_ini_str = data_inci.strftime('%Y-%m-%d') #Posar les dades en format Earth Engine (YYYY-MM-DD)
            data_fin_str = data_final.strftime('%Y-%m-%d')#Posar les dades en format Earth Engine (YYYY-MM-DD)

        #Netejem la carpeta abans de descarregar les imatges noves:
        with st.spinner("🧹 Netejant imatges de proves anteriors..."):
            if os.path.exists(ruta_carpeta):
                # Recorrem tots els arxius de la carpeta i esborrem els .tif
                for arxiu_vell in os.listdir(ruta_carpeta):
                    if arxiu_vell.endswith('.tif'):
                        os.remove(os.path.join(ruta_carpeta, arxiu_vell))
            else:
                # Si la carpeta no existeix, la creem perquè no doni error
                os.makedirs(ruta_carpeta)


        #Cridem la funció d'extracció i li passem les dades dinàmiques
        with st.spinner("🌍 Connectant amb el satèl·lit i descarregant imatges..."):
            exit_descarrega = data_extraction_2.extreure_imatges_satelit(
                bbox =  bbox_calculat,
                data_ini=data_ini_str,
                data_fin=data_fin_str,
                dir_sortida = ruta_carpeta,
                mode_historic = historic_activat
            )
        #st.spinner és una animació de càrrega, pq connectarse a Google Earth i descarregar les imatges triga uns segons
        #with és per gestionar contextos



            
        if exit_descarrega:
            with st.spinner("Processant imatges a la memòria... "):
                resultats = gpu_processing.processar_directori(ruta_carpeta)
            st.session_state['resultats_processats'] = resultats
            st.success("Processament completat amb èxit! Desplaça't cap avall per veure'n els resultats.")
        else:
            st.error(f"No s'han trobat imatges vàlides o ha fallat la descàrrega.")


if 'resultats_processats' in st.session_state:
    resultats = st.session_state['resultats_processats']

    col_esq, col_drt = st.columns([1.3, 1]) #Per ajustar grandaria de part dreta i esquerra una vegada processades les imategs (dels resultats)
    with col_esq:
        st.subheader("📊 Registre d'Observacions:")

        for i in resultats:
            #DESPLEGABLE PER CADA IMATGE
            #Afegim el percentatge de núvols al desplegable
            with st.expander(f"📅 Data: {i['data']}  |  💧 {i['hectarees']:.2f} ha |  ☁️ Núvols: {i['perc_nuvols']:.1f}%"):

                c1,c2,c3 = st.columns(3)

                with c1:
                    ruta_rgb = os.path.join(ruta_carpeta, i['rgb_png'])
                    if os.path.exists(ruta_rgb):
                        st.image(ruta_rgb, caption="1. Vista Real (Satèl·lit)", use_container_width=True)
                        
                with c2:
                    ruta_cloud = os.path.join(ruta_carpeta, i['cloud_png'])
                    if os.path.exists(ruta_cloud):
                        st.image(ruta_cloud, caption=f"2. Detecció de Núvols (AI Mask: {i['perc_nuvols']:.1f}%)", use_container_width=True)
                        
                with c3:
                    ruta_png_real = os.path.join(ruta_carpeta, i['imatge_png'])
                    if os.path.exists(ruta_png_real):
                        # AQUÍ ESTÀ LA SOLUCIÓ: Cridem a 'ruta_png_real' i canviem el títol
                        st.image(ruta_png_real, caption="3. Detecció d'Aigua (NDWI)", use_container_width=True)

                # --- NOU: Destaquem el percentatge de núvols a sota ---
                st.markdown("---") # Línia separadora
                # Utilitzem st.metric que és una eina de Streamlit per mostrar dades clau de forma grossa i professional
                col_metrica1, col_metrica2 = st.columns(2)
                with col_metrica1:
                    st.metric(label="☁️ Cobertura de Núvols (Detectat amb IA):", value=f"{i['perc_nuvols']:.2f} %")
                with col_metrica2:
                    st.metric(label="💧 Superfície d'Aigua:", value=f"{i['hectarees']:.2f} ha")
                
                st.markdown("---")
#

                st.write(f"**Nom original: ** '{i['arxiu']}'")
                st.write(f"**Estat:** Processat correctament a la GPU.")

                ruta_imatge_real = os.path.join(ruta_carpeta, i['arxiu'])

                with open(ruta_imatge_real, "rb") as file:
                    st.download_button(
                        label="📥 Descarregar Imatge Real de Satèl·lit (.tif)",
                        data=file,
                        file_name=i['arxiu'],
                        mime="image/tiff",
                        key=i['arxiu'] # Clau única obligatòria perquè Streamlit no es confongui de botó
                    )
                st.info("💡 Nota: Els arxius .tif multispectrals requereixen programari GIS (com QGIS) per a la seva visualització.")
    with col_drt:
        st.subheader("📈 Evolució de la superfície d'aigua: ")
                            
        # Cridem a la teva funció. Com que estem dins del "with col_drt", 
        # Streamlit dibuixarà la gràfica automàticament a la dreta!
        analisis.generar_grafic_evolucio(resultats)

        st.write("---")
        st.subheader(" 🎥 Timelapse Satel·lital: ")

        #Definim on es guardarà l 'arxiu de vídeo
        ruta_gif_final = os.path.join(ruta_carpeta, "timelapse_sau.gif")

        with st.spinner("Building satellite timelapse over time..."):
            exit_gif = analisis.generar_timelapse(resultats, ruta_carpeta, ruta_gif_final)

            if exit_gif:
                st.image(ruta_gif_final, use_container_width=True)

            else:
                st.warning("Timelapse couldn't be generated.")
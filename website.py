import streamlit as st
import os
import matplotlib.pyplot as plt
import folium #per poder posar un mapa
from streamlit_folium import st_folium
from folium.plugins import Draw
import datetime
import shutil #ens permet esborrar arxius
import sys
import time
import math #per calcular la mida aproximada de la zona (km2) i avisar si és massa gran per descarregar

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
    # Si falla, l'excepció ha de sortir d'aquí: st.cache_resource no cacheja les funcions que llancen error,
    # així es torna a intentar a la següent execució (si capturéssim l'error aquí, es cachejaria el fracàs).
    data_extraction_2.inicialitzar_gee()

try:
    iniciar_connexio_satelit()
except Exception as e:
    st.error(f"Error connectant a Earth Engine: {e}")


#------------------------------------------------

# Llegim l'estat del toggle de mode incendis ABANS de crear-lo com a widget (més avall), perquè
# st.set_page_config() ha de ser la primera crida de Streamlit i necessitem saber el mode per triar
# el títol. session_state ja té el valor actualitzat d'una interacció anterior (si n'hi ha hagut); la
# primera vegada que es carrega la pàgina, la clau encara no existeix i per defecte és fals (mode aigua).
incendi_actiu = st.session_state.get("mode_incendi_toggle", False)

st.set_page_config(
    page_title = "Wildfire Detection" if incendi_actiu else "Water Reservoir Monitoring",
    page_icon = "🔥" if incendi_actiu else "💧",
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
        font-weight: bold !important; /* Això posa la lletra en negreta */
    }
    
    /* 2. Fer el quadradet de la casella més gran perquè quedi compensat */
    div[data-testid="stCheckbox"] div[role="checkbox"] {
        transform: scale(1); /* Augmenta la mida del quadrat un 50% */
        margin-right: 15px; /* Deixa una mica d'aire entre el quadrat i el text */
        margin-left: 5px;
    }
    /* --- REDUIR L'ESPAI BLANC SUPERIOR --- */
    .block-container {
        padding-top: 1rem !important; /* Redueix el marge de dalt. Per defecte és 6rem. Es pot canviar: 2rem, 1rem, 0rem... */
        padding-bottom: 1rem !important; /* Redueix el marge de baix per si de cas */
    }

    /* --- CENTRAR LES MÈTRIQUES DE NÚVOLS I AIGUA --- */
    div[data-testid="stMetric"] {
        text-align: center !important;
    }
    div[data-testid="stMetricValue"] > div {
        justify-content: center !important;
    }
</style>
""", unsafe_allow_html = True)

if incendi_actiu:
    st.title("On-board Satellite processing: Wildfire detection")
    st.write("This application simulates onboard GPU processing on a satellite to detect and track wildfire growth in near real time.")
else:
    st.title("On-board Satellite processing: Water reservoir monitoring")
    st.write("This application simulates data processing with GPU's on a satellite and shows water evolution over time.")

# La demo serveix per mostrar l'ús de GPUs: si no hi ha CUDA, que no passi desapercebut que va en CPU
if not gpu_processing.gpu_disponible():
    st.warning(
        "⚠️ **No s'ha detectat cap GPU NVIDIA (CUDA).** El processament s'executarà en **CPU** i els temps "
        "mesurats NO serveixen per demostrar l'acceleració per GPU. Si això és el servidor, comprova que el "
        "contenidor s'hagi engegat amb `--gpus all` i que el driver sigui compatible amb la versió de CUDA de PyTorch."
    )

# 1 Carpeta de dades interna del servidor (L'usuari no la veu)
ruta_carpeta = os.path.join(os.getcwd(), 'dades_satelit_temporals')


# --- MODE INCENDIS: cas d'ús de resposta en temps crític ---
# El posem aquí dalt (abans del mapa i les dates) perquè tota la resta de la pàgina en depèn:
# amb quines bandes es descarrega, quina zona surt per defecte al mapa, i quin processament s'executa.
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔥 OBP Edge Computing")
mode_incendi = st.sidebar.toggle("Activar Mode Incendis (resposta en temps crític)", value=False, key="mode_incendi_toggle")

# Dos casos reals validats amb dades de Sentinel-2. La Bisbal és el cas per defecte: tot l'incendi
# cap en un sol parell abans/després (24-48h), amb una comparació molt neta. Sierra Oeste és més gran
# però es propaga durant setmanes, útil per ensenyar una sèrie de creixement més llarga.
ESCENARIS_INCENDI = {
    "🌲 Gavarres, La Bisbal d'Empordà (jul 2026)": {
        "center": [41.885, 3.05], "zoom": 10,
        "bbox": [2.90, 41.79, 3.10, 41.98], # validat: conté tota la taca cremada (91% de les 2.200 ha oficials)
        "bbox_graella": [2.90, 41.79, 3.30, 41.98], # doble d'ample: la meitat esquerra és l'incendi, la dreta és zona de control
        "inici": datetime.date(2026, 6, 25), "final": datetime.date(2026, 7, 10),
        "tile": "T31TDG", # la zona cau al solapament amb T31TEG; fixem la tessel·la per no duplicar dates
        "descripcio": (
            "🔥 **Cas real: Incendi de les Gavarres (La Bisbal d'Empordà), 3-4 de juliol de 2026** — "
            "~2.130 ha de bosc cremades en només 24-48 hores. Ja hi ha un rectangle preseleccionat que "
            "conté tota la taca cremada; pots dibuixar-ne un altre si vols provar una altra àrea. "
            "Comparació abans/després molt neta: imatge del 25/06 (abans) i del 05/07 (2 dies després)."
        ),
    },
    "⛰️ Sierra Oeste, Madrid/Àvila (jul-ago 2026)": {
        "center": [40.32, -4.47], "zoom": 11,
        "bbox": [-4.50, 40.28, -4.38, 40.37], # validat
        "bbox_graella": [-4.62, 40.28, -4.38, 40.37], # doble d'ample cap a l'oest
        "inici": datetime.date(2026, 7, 20), "final": datetime.date(2026, 8, 5),
        "tile": "T30TUK", # evita duplicats per orbites que se superposen en algunes dates
        "descripcio": (
            "🔥 **Cas real: Incendi de la Sierra Oeste (Madrid/Àvila), juliol-agost 2026** — un dels "
            "més grans de la història de la zona, propagat durant més de dues setmanes. Rectangle "
            "preseleccionat sobre una part de la zona afectada (l'incendi complet és molt més gran)."
        ),
    },
}

if mode_incendi:
    nom_escenari = st.sidebar.selectbox("Selecciona l'incendi:", list(ESCENARIS_INCENDI.keys()))
    escenari_incendi = ESCENARIS_INCENDI[nom_escenari]
    mode_graella = st.sidebar.checkbox(
        "🔲 Simular escaneig per satèl·lit (graella)",
        value=False,
        help=(
            "En lloc de processar només el requadre exacte de l'incendi (que ja el sabem, no l'hem "
            "\"descobert\"), divideix una zona més gran en tessel·les i les processa una per una, com "
            "faria un satèl·lit escanejant una franja de terreny sense saber a priori on hi ha res "
            "interessant. Cada tessel·la es classifica per separat: prioritat alta (canvi detectat) o "
            "sense canvi (es descarta)."
        )
    )
else:
    nom_escenari = None
    escenari_incendi = None
    mode_graella = False

if mode_incendi:
    with st.sidebar.expander("ℹ️ Per què aquest mode és diferent del d'embassaments", expanded=False):
        st.markdown(
            "**Embassaments (mode normal):** és la prova de concepte — demostra que es pot fer "
            "processament amb GPU a l'espai. El temps de resposta no és crític: un embassament no canvia "
            "gaire d'una passada del satèl·lit a la següent (uns dies), així que processar-lo més ràpid "
            "o més lent amb GPU no canvia res a la pràctica.\n\n"
            "**Incendis (aquest mode):** aquí sí que importa la velocitat, però no en el sentit de "
            "\"detectar-ho abans que una persona vegi el fum\" — amb un satèl·lit òptic que passa cada "
            "2-5 dies, això és físicament impossible, i seria enganyós prometre-ho. El valor real és un "
            "altre: quan finalment arriba una passada útil, la GPU decideix a l'instant si aquella imatge "
            "mostra un canvi prou important per prioritzar-ne la baixada, en lloc de baixar-ho tot "
            "cegament i que algú a terra ho miri hores o dies després. A escala d'una constel·lació "
            "sencera vigilant un territori gran, això només és viable amb processament paral·lel (GPU), "
            "no amb una CPU feble fent-ho en sèrie dins el pressupost de potència d'un satèl·lit."
        )
    st.sidebar.warning(
        "Mode actiu: es descarreguen bandes addicionals (SWIR, B11/B12) per calcular l'índex de "
        "cremat (NBR) i es prioritzen les imatges amb creixement significatiu respecte a l'última passada útil."
    )


#----------------------------------------------------------------------------------------------------------
#AFEGIR UNA SELECCIÓ DE DATES:
st.write("🗓️ Selecciona l'interval de temps:")
col_data1, col_data2 = st.columns(2) #Creem dues columnes

data_minima_s2 = datetime.date(2017,1,1)
data_avui = datetime.date.today()

# El mode incendis no té sentit amb el gràfic històric (és un cas d'estudi puntual, no una evolució d'anys)
historic_activat = st.checkbox("📊 Generar gràfic històric complet (2017- Avui)", disabled = mode_incendi)

# Dates per defecte: les del cas real seleccionat (ESCENARIS_INCENDI) en mode incendis; Sau normalment
# en mode aigua. La key inclou l'escenari perquè Streamlit consideri que és un widget "nou" en canviar
# de mode o d'incendi, i apliqui el nou value per defecte.
if mode_incendi:
    valor_inici_defecte = escenari_incendi["inici"]
    valor_final_defecte = escenari_incendi["final"]
else:
    valor_inici_defecte = datetime.date(2026,1,1)
    valor_final_defecte = datetime.date.today()

#quan posme with li estem dient a python que tot el que posem dins del with volem que ho posi dins de la columna
with col_data1:
    data_inci = st.date_input(
        "Data d'inici",
        value = valor_inici_defecte,
        min_value = data_minima_s2,
        max_value = data_avui,
        disabled = historic_activat,
        key = f"data_inici_{mode_incendi}_{nom_escenari}"
        )

with col_data2:
    data_final = st.date_input(
        "Data final",
        value = valor_final_defecte,
        min_value = data_minima_s2,
        max_value= data_avui,
        disabled = historic_activat,
        key = f"data_final_{mode_incendi}_{nom_escenari}"
        )


# --- MAPA INTERACTIU I DINÀMIC ---
st.subheader("📍 Selecciona l'àrea d'interès del mapa:")

# Coordenades aproximades del Pantà de Sau
latitud_sau = 41.986
longitud_sau = 2.398

if mode_incendi:
    st.info(escenari_incendi["descripcio"] + " Pots dibuixar un altre rectangle si vols provar una altra àrea.")
    map_center = escenari_incendi["center"]
    map_zoom = escenari_incendi["zoom"]
else:
    # Mode normal: Amaguem el selector i anem directes a Sau
    st.info("ℹ️ Navegació lliure: Desplaça't pel mapa o dibuixa la zona a monitoritzar.")
    map_center = [latitud_sau, longitud_sau]
    map_zoom = 8

# Si l'usuari canvia de mode, d'incendi o del checkbox de graella, el rectangle dibuixat abans ja no
# és a la vista del mapa: l'oblidem perquè no es processi una zona "fantasma" que l'usuari no veu. En
# mode incendis, en lloc de deixar-ho buit, preseleccionem el rectangle validat de l'escenari (bbox
# normal, o bbox_graella si el mode escaneig està actiu). L'usuari pot dibuixar-ne un altre si vol;
# el seu dibuix sempre substitueix aquest per defecte.
clau_zona = f"{mode_incendi}_{nom_escenari}_{mode_graella}"
bbox_defecte = None
if mode_incendi:
    bbox_defecte = escenari_incendi.get("bbox_graella") if mode_graella else escenari_incendi.get("bbox")

if st.session_state.get('clau_zona') != clau_zona:
    if bbox_defecte:
        lon_min, lat_min, lon_max, lat_max = bbox_defecte
        st.session_state['coordenades_guardades'] = [
            [lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]
        ]
    else:
        st.session_state.pop('coordenades_guardades', None)
    st.session_state['clau_zona'] = clau_zona

# Creem el mapa centrat a Sau amb Folium (o a la zona d'emergència)
#Mapa per defecte: m = folium.Map(location=[latitud_sau, longitud_sau], zoom_start=12)
# 1. Crear el mapa amb l'eina de dibuix activada (Amb capa de satèl·lit de Google)
m = folium.Map(
    location=map_center, 
    zoom_start=map_zoom, # He allunyat una mica el zoom per veure més territori (Mode normal)
    tiles="http://mt0.google.com/vt/lyrs=y&hl=ca&x={x}&y={y}&z={z}",
    #posar lyrs=y si volem que surtin noms + satèlit
    #lyrs=s si volem que sigui només imatge de satèl·lit
    attr="Google"
)

# En mode incendis, dibuixem el rectangle preseleccionat perquè l'usuari el vegi (no és invisible)
if mode_incendi and bbox_defecte:
    lon_min, lat_min, lon_max, lat_max = bbox_defecte
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="#28a745", weight=3, fill=True, fill_opacity=0.08,
        tooltip="Zona preseleccionada (pots dibuixar-ne una altra a sobre)"
    ).add_to(m)

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
#key: el mapa es torna a crear (i s'esborren els dibuixos) quan canvia el mode o l'escenari
output_mapa = st_folium(m, height=500, use_container_width=True, returned_objects =["all_drawings"], key=f"mapa_{clau_zona}")

coordenades_rectangle = None

# Botó per si l'usuari ha mogut/esborrat sense voler el rectangle preseleccionat (p.ex. amb l'eina
# d'editar/esborrar del mapa) i vol tornar a la zona validada, sense haver de canviar d'escenari.
if mode_incendi and bbox_defecte:
    if st.button("🔄 Restaurar zona preseleccionada", key="restaurar_zona"):
        lon_min, lat_min, lon_max, lat_max = bbox_defecte
        st.session_state['coordenades_guardades'] = [
            [lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]
        ]
        st.rerun()
    # Mostrem sempre quina és la zona activa ara mateix, per no haver d'endevinar-ho mirant el mapa
    zona_activa = st.session_state.get('coordenades_guardades')
    if zona_activa:
        lons_act = [p[0] for p in zona_activa]; lats_act = [p[1] for p in zona_activa]
        st.caption(
            f"📍 Zona activa ara mateix: ({min(lons_act):.3f}, {min(lats_act):.3f}) → "
            f"({max(lons_act):.3f}, {max(lats_act):.3f})"
        )

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
    boto_executat = st.button("🚀 Executar Processament (GPU)", width='stretch')


if boto_executat:
    coordenades_rectangle = st.session_state.get('coordenades_guardades', None)
    #Comprovem si l'usuari ha adibuixat un rectangle
    if coordenades_rectangle is None:
        st.warning("⚠️ Si us plau, dibuixa un rectangle al mapa abans de començar el processament.")

    elif not historic_activat and data_inci > data_final:
        st.warning("⚠️ La data d'inici no pot ser posterior a la data final.")

    else:
        #Convertir el polígon de Folium a BBOX (lon_min, lat_min, lon_max, lat_max)
        lons=[punt[0] for punt in coordenades_rectangle]
        lats = [punt[1] for punt in coordenades_rectangle]
        bbox_calculat = [min(lons), min(lats), max(lons), max(lats)]

        n_bandes_previst = len(data_extraction_2.BANDES_FOC) if mode_incendi else len(data_extraction_2.BANDES_AIGUA)
        # En mode incendis fem servir escala 20m (B11/B12 ja són natives a 20m a Sentinel-2, no es perd
        # resolució real) en lloc de 10m: quadruplica l'àrea que cap en una sola imatge de 48 MB.
        escala_previst = 20 if mode_incendi else 10

        if historic_activat:
            data_ini_str = '2017-03-28'
            data_fin_str = data_avui.strftime('%Y-%m-%d')#Posar les dades en format Earth Engine (YYYY-MM-DD)
        else:
            data_ini_str = data_inci.strftime('%Y-%m-%d') #Posar les dades en format Earth Engine (YYYY-MM-DD)
            #filterDate d'Earth Engine exclou la data final: hi sumem un dia perquè el dia triat quedi inclòs
            data_fin_str = (data_final + datetime.timedelta(days=1)).strftime('%Y-%m-%d')#Posar les dades en format Earth Engine (YYYY-MM-DD)

        # --- Comprovació de mida: Earth Engine limita cada imatge descarregada a ~48 MB ---
        # Amb més bandes (mode incendis: 6, per calcular el NBR) aquest límit s'assoleix amb una zona
        # més petita que en mode aigua (4 bandes). Ho comprovem ABANS de descarregar res: si no,
        # geemap salta cada imatge en silenci (només ho diu per consola) i l'app acaba mostrant
        # "no s'han trobat imatges", amagant que la causa real és la mida del rectangle.
        # En mode graella NO cal: ja dividim la zona en tessel·les prou petites (vegeu dividir_en_graella).
        mida_excessiva = False
        if not mode_graella:
            LIMIT_BYTES_PER_IMATGE = data_extraction_2.LIMIT_BYTES_PER_IMATGE
            MARGE_SEGURETAT = data_extraction_2.MARGE_SEGURETAT_MIDA
            lat_mitjana = (bbox_calculat[1] + bbox_calculat[3]) / 2
            amplada_m = (bbox_calculat[2] - bbox_calculat[0]) * 111320 * math.cos(math.radians(lat_mitjana))
            alcada_m = (bbox_calculat[3] - bbox_calculat[1]) * 111320
            bytes_estimats = (amplada_m / escala_previst) * (alcada_m / escala_previst) * n_bandes_previst * 4  # float32

            if bytes_estimats > LIMIT_BYTES_PER_IMATGE * MARGE_SEGURETAT:
                mida_excessiva = True
                costat_maxim_km = data_extraction_2.calcular_costat_maxim_km(n_bandes_previst, escala_previst, MARGE_SEGURETAT)
                st.warning(
                    f"⚠️ La zona triada és massa gran per descarregar-la sencera d'un sol cop: amb "
                    f"{n_bandes_previst} bandes, Google Earth Engine limita cada imatge a ~48 MB. "
                    f"Dibuixa un rectangle més petit (aproximadament {costat_maxim_km:.0f}×{costat_maxim_km:.0f} km "
                    f"com a màxim si és quadrat; menys, si és més allargat), o activa el mode escaneig "
                    f"per graella perquè es divideixi automàticament."
                )

        if not mida_excessiva:
            #Netejem la carpeta abans de descarregar les imatges noves:
            with st.spinner("🧹 Netejant imatges de proves anteriors..."):
                if os.path.exists(ruta_carpeta):
                    shutil.rmtree(ruta_carpeta)
                os.makedirs(ruta_carpeta)

            #Cridem la funció d'extracció i li passem les dades dinàmiques
            with st.expander("Terminal de processament en directe:", expanded=True):
                terminal_web = st.empty()

            class CapturadorConsola:
                def __init__(self):
                    self.contingut = "> Inicialitzant procés al servidor Caos17...\n"
                    terminal_web.code(self.contingut, language='bash')

                def write(self, text):
                    try:
                        sys.__stdout__.write(text) # Que surti també al 'docker logs' original
                    except UnicodeEncodeError:
                        # Consola que no és UTF-8 (p. ex. Windows amb cp1252): els emojis no han de tombar el procés
                        codificacio = getattr(sys.__stdout__, 'encoding', None) or 'ascii'
                        sys.__stdout__.write(text.encode(codificacio, errors='replace').decode(codificacio))
                    if text.strip() and not text.isspace(): # Neteja línies buides
                        self.contingut += text.strip() + "\n"
                        terminal_web.code(self.contingut, language='bash')

                def flush(self):
                    sys.__stdout__.flush()

            canal_original = sys.stdout
            sys.stdout = CapturadorConsola()

            start_time = time.time()

            try:
                bandes_a_descarregar = data_extraction_2.BANDES_FOC if mode_incendi else None
                # Mode incendis: fixem la tessel·la de l'escenari (si en té) perquè no es dupliqui cada
                # data en zones de solapament entre tessel·les de Sentinel-2.
                tile_a_filtrar = escenari_incendi.get("tile") if mode_incendi else None

                if mode_graella:
                    # ============================================================================
                    # MODE ESCANEIG PER GRAELLA: simulem un satèl·lit escanejant una franja de
                    # terreny SENSE que ningú li digui on és l'incendi. Dividim la zona en tessel·les
                    # (cadascuna dins del límit de mida de Earth Engine) i les processem una per una,
                    # exactament igual que faríem amb una sola zona, però repetit tessel·la a tessel·la.
                    # ============================================================================
                    tessel_les, n_files, n_cols = data_extraction_2.dividir_en_graella(
                        bbox_calculat, n_bandes_previst, escala_previst
                    )
                    print(f"🔲 Dividint la zona en una graella de {n_files}×{n_cols} = {len(tessel_les)} tessel·les.")
                    print("El satèl·lit escanejarà cada tessel·la per separat, sense saber a priori on hi ha res d'interessant.\n")

                    resultats_graella = []
                    with st.spinner(f"🛰️ Escanejant {len(tessel_les)} tessel·les..."):
                        for t in tessel_les:
                            print(f"--- Tessel·la fila {t['fila']}, columna {t['col']} ---")
                            carpeta_tile = os.path.join(ruta_carpeta, f"tile_{t['fila']}_{t['col']}")
                            os.makedirs(carpeta_tile, exist_ok=True)

                            # NO reutilitzem tile_a_filtrar (la tessel·la fixa de l'escenari): la graella
                            # s'estén més enllà del requadre validat, i cada cel·la pot caure sobre una
                            # tessel·la real diferent de Sentinel-2. Detectem quina hi predomina abans de
                            # descarregar res: sense això, o forçant-ne una que no cobreix la cel·la,
                            # acabàvem barrejant dues tessel·les diferents i sortia un "cremat" fals a la
                            # vora entre totes dues (comprovat amb dades reals).
                            tile_cella = data_extraction_2.tessel_la_predominant(t['bbox'], data_ini_str, data_fin_str)
                            ok_tile = data_extraction_2.extreure_imatges_satelit(
                                bbox=t['bbox'], data_ini=data_ini_str, data_fin=data_fin_str,
                                dir_sortida=carpeta_tile, bandes=bandes_a_descarregar,
                                tile=tile_cella, scale=escala_previst
                            )
                            res_tile = gpu_processing.processar_directori_incendi(carpeta_tile)[0] if ok_tile else []

                            if res_tile:
                                ultim_tile = res_tile[-1]
                                # Llindar propi del mode graella (LLINDAR_GRAELLA_HA), NO el mateix que
                                # LLINDAR_CREIXEMENT_ALERTA: aquell és per detectar un salt d'un dia a
                                # l'altre en una zona ja coneguda; aquí cal separar un incendi real del
                                # soroll normal de fons d'una tessel·la sense res d'interessant.
                                classificacio = ('prioritat_alta'
                                                 if ultim_tile['hectarees_cremades'] > gpu_processing.LLINDAR_GRAELLA_HA
                                                 else 'sense_canvi')
                                ruta_rgb = os.path.join(carpeta_tile, ultim_tile['rgb_png'])
                                resultats_graella.append({
                                    'fila': t['fila'], 'col': t['col'], 'bbox': t['bbox'],
                                    'hectarees_cremades': ultim_tile['hectarees_cremades'],
                                    'classificacio': classificacio,
                                    'rgb_png': ruta_rgb if os.path.exists(ruta_rgb) else None,
                                })
                                print(f"  -> {classificacio} ({ultim_tile['hectarees_cremades']:.1f} ha)\n")
                            else:
                                resultats_graella.append({
                                    'fila': t['fila'], 'col': t['col'], 'bbox': t['bbox'],
                                    'hectarees_cremades': None, 'classificacio': 'sense_dades', 'rgb_png': None,
                                })
                                print("  -> sense dades útils (cap imatge vàlida en aquesta tessel·la)\n")

                    temps_total = round(time.time() - start_time, 2)
                    n_prioritat = sum(1 for r in resultats_graella if r['classificacio'] == 'prioritat_alta')

                    st.session_state['resultats_graella'] = resultats_graella
                    st.session_state['graella_forma'] = (n_files, n_cols)
                    st.session_state['mode_resultats'] = 'graella'
                    st.session_state.pop('resultats_processats', None)
                    st.success(f"Escaneig completat: {len(tessel_les)} tessel·les processades en {temps_total} s.")
                    st.info(f"🔥 **{n_prioritat} de {len(tessel_les)}** tessel·les marcades amb prioritat de baixada ALTA.")

                else:
                    # ============================================================================
                    # FLUX D'UNA SOLA ZONA (aigua o incendi amb sèrie temporal detallada)
                    # ============================================================================
                    with st.spinner("🌍 Connectant amb el satèl·lit i descarregant imatges..."):
                        exit_descarrega = data_extraction_2.extreure_imatges_satelit(
                            bbox =  bbox_calculat,
                            data_ini=data_ini_str,
                            data_fin=data_fin_str,
                            dir_sortida = ruta_carpeta,
                            mode_historic = historic_activat,
                            bandes = bandes_a_descarregar,
                            tile = tile_a_filtrar,
                            scale = escala_previst
                        )
                    #st.spinner és una animació de càrrega, pq connectarse a Google Earth i descarregar les imatges triga uns segons
                    #with és per gestionar contextos

                    if exit_descarrega:
                        with st.spinner("Processant imatges a la memòria... "):
                            if mode_incendi:
                                resultats, temps_gpu_total, temps_cpu_total, bytes_totals, n_descartades = gpu_processing.processar_directori_incendi(ruta_carpeta)
                            else:
                                resultats, temps_gpu_total, temps_cpu_total = gpu_processing.processar_directori(ruta_carpeta)
                                bytes_totals = None
                                n_descartades = None

                        temps_total = round(time.time() - start_time, 2)
                        nom_gpu, mem_gpu = gpu_processing.obtenir_estadistiques_hardware()

                        st.session_state['resultats_processats'] = resultats
                        st.session_state['mode_resultats'] = 'incendi' if mode_incendi else 'aigua'
                        st.session_state['bytes_totals_resultats'] = bytes_totals
                        st.session_state['temps_gpu_resultats'] = temps_gpu_total
                        st.session_state['temps_cpu_resultats'] = temps_cpu_total
                        st.session_state['n_descartades_resultats'] = n_descartades
                        st.session_state.pop('resultats_graella', None)
                        st.session_state.pop('clau_gif', None) # nou processament => cal regenerar el timelapse
                        st.success("Processament completat amb èxit!")

                        # Mostrar a la web
                        st.info(
                            f"**Rendiment Global:** Total: {temps_total} s  |  "
                            f" **GPU :** {temps_gpu_total} s  |  "
                            f" **CPU (In development):** {temps_cpu_total} s\n\n"
                            f"**Hardware:** {nom_gpu}  |  **VRAM:** {mem_gpu} GB"
                        )

                    else:
                        # Mostrem la zona i dates EXACTES que s'han intentat, perquè es pugui saber si
                        # el rectangle no és el que es pensava (p.ex. mogut per accident amb l'eina
                        # d'editar del mapa) en lloc de només dir "no s'han trobat imatges".
                        st.error(
                            "No s'han trobat imatges vàlides o ha fallat la descàrrega.\n\n"
                            f"**Zona intentada:** ({bbox_calculat[0]:.3f}, {bbox_calculat[1]:.3f}) → "
                            f"({bbox_calculat[2]:.3f}, {bbox_calculat[3]:.3f})  |  "
                            f"**Dates:** {data_ini_str} a {data_fin_str}"
                            + (f"  |  **Tessel·la:** {tile_a_filtrar}" if tile_a_filtrar else "") +
                            "\n\nSi el rectangle és gran, Google Earth Engine pot rebutjar cada imatge per "
                            "superar el límit de mida (~48 MB/imatge); prova amb una zona més petita, o "
                            "restaura la zona preseleccionada amb el botó de sobre del mapa."
                        )

            finally:
                sys.stdout = canal_original


# ==========================================================================================
# MODE ESCANEIG PER GRAELLA: resultats separats (no és una sèrie temporal d'una sola zona)
# ==========================================================================================
if st.session_state.get('mode_resultats') == 'graella' and 'resultats_graella' in st.session_state:
    resultats_graella = st.session_state['resultats_graella']
    n_files, n_cols = st.session_state['graella_forma']
    n_prioritat = sum(1 for r in resultats_graella if r['classificacio'] == 'prioritat_alta')

    st.subheader("🛰️ Resultat de l'escaneig per satèl·lit")
    st.write(
        "Cada quadre és una tessel·la processada de forma independent, com faria un satèl·lit "
        "escanejant una franja de terreny sense saber a priori on hi ha res d'interessant. "
        "🟢 verd = sense canvi (es descartaria); 🔴 vermell = prioritat de baixada alta (canvi "
        "significatiu, possible incendi); ⚪ gris = sense dades útils (massa núvols o sense imatges)."
    )
    col_metrica1, col_metrica2 = st.columns(2)
    with col_metrica1:
        st.metric("Tessel·les processades", len(resultats_graella))
    with col_metrica2:
        st.metric("Prioritat ALTA", f"{n_prioritat} / {len(resultats_graella)}")

    analisis.generar_graella_escaneig(resultats_graella, n_files, n_cols)

    with st.expander("Detall de cada tessel·la"):
        for r in resultats_graella:
            ha_text = f"{r['hectarees_cremades']:.1f} ha" if r['hectarees_cremades'] is not None else "—"
            st.write(f"**Fila {r['fila']}, columna {r['col']}** ({r['bbox'][0]:.3f}, {r['bbox'][1]:.3f}) → "
                     f"({r['bbox'][2]:.3f}, {r['bbox'][3]:.3f})  |  {r['classificacio']}  |  {ha_text}")

    st.info(
        "💡 A diferència del mode d'una sola zona, aquí NO li hem dit al sistema on és l'incendi: "
        "ha calgut escanejar totes les tessel·les per trobar-lo. Aquesta és la part que demostra el "
        "valor de la GPU a bord: descartar en segons les tessel·les sense interès, en lloc de baixar-les "
        "totes senceres i que algú les miri una per una a terra."
    )


if 'resultats_processats' in st.session_state:
    resultats = st.session_state['resultats_processats']
    mode_resultats = st.session_state.get('mode_resultats', 'aigua')

    col_esq, col_drt = st.columns([1.3, 1]) #Per ajustar grandaria de part dreta i esquerra una vegada processades les imategs (dels resultats)

    # ======================================================================================
    # MODE INCENDIS: registre + gràfic de superfície cremada
    # ======================================================================================
    if mode_resultats == 'incendi':
        with col_esq:
            st.subheader("🔥 Registre d'Observacions (Incendi):")

            # --- Resum de la missió: cop d'ull a l'operació sencera abans d'entrar en detall ---
            n_descartades_disp = st.session_state.get('n_descartades_resultats')
            bytes_totals_disp = st.session_state.get('bytes_totals_resultats') or 0
            n_prioritat_disp = sum(1 for r in resultats if r.get('alerta_creixement', False))
            total_captades_disp = len(resultats) + (n_descartades_disp or 0)

            st.markdown("#### 🛰️ Resum de la missió")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Passades captades", total_captades_disp)
            with col_m2:
                st.metric("Descartades (núvols)", n_descartades_disp if n_descartades_disp is not None else "—")
            with col_m3:
                st.metric("Prioritat ALTA", n_prioritat_disp)
            with col_m4:
                st.metric("Dades processades", f"{bytes_totals_disp/1024/1024:.0f} MB")

            # Línia de temps compacta: què ha decidit el sistema a CADA passada, d'un cop d'ull.
            # Les descartades per núvols no hi surten (s'han esborrat i no en sabem la data), però la
            # xifra de dalt ("Descartades") ja indica que n'hi ha hagut.
            st.markdown("**🧭 Seqüència de decisions a bord:**")
            icones = []
            for r in resultats:
                if r.get('es_referencia', False):
                    icones.append(f"📌 {r['data']}")
                elif r.get('alerta_creixement', False):
                    icones.append(f"🔥 {r['data']}")
                else:
                    icones.append(f"✅ {r['data']}")
            st.write("  →  ".join(icones))
            st.caption("📌 referència  |  🔥 prioritat de baixada alta (canvi significatiu)  |  "
                       "✅ passada vàlida, sense canvi significatiu")
            st.markdown("---")

            # --- Comparació ABANS / DESPRÉS: cop d'ull ràpid amb la referència i l'última observació ---
            referencia = next((r for r in resultats if r.get('es_referencia', False)), None)
            ultim = resultats[-1] if resultats else None
            if referencia and ultim and referencia is not ultim:
                st.markdown("#### 📸 Comparació abans / després")
                col_abans, col_despres = st.columns(2)
                with col_abans:
                    st.markdown(f"**🟢 ABANS — {referencia['data']}**")
                    ruta = os.path.join(ruta_carpeta, referencia['rgb_png'])
                    if os.path.exists(ruta):
                        st.image(ruta, width='stretch')
                with col_despres:
                    st.markdown(f"**🔴 DESPRÉS — {ultim['data']}**")
                    ruta = os.path.join(ruta_carpeta, ultim['rgb_png'])
                    if os.path.exists(ruta):
                        st.image(ruta, width='stretch')
                st.metric(
                    label=f"🔥 Superfície cremada detectada ({referencia['data']} → {ultim['data']})",
                    value=f"{ultim['hectarees_cremades']:.1f} ha"
                )
                st.markdown("---")

            # --- Perímetre cremat SOBRE EL MAPA: no només com a imatge solta, sinó situat geogràficament ---
            if ultim and ultim.get('overlay_png') and ultim.get('limits_geo'):
                ruta_overlay = os.path.join(ruta_carpeta, ultim['overlay_png'])
                if os.path.exists(ruta_overlay):
                    st.markdown("#### 🗺️ Perímetre cremat sobre el mapa")
                    lat_min, lon_min, lat_max, lon_max = ultim['limits_geo']
                    mapa_incendi = folium.Map(
                        location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                        zoom_start=12,
                        tiles="http://mt0.google.com/vt/lyrs=y&hl=ca&x={x}&y={y}&z={z}",
                        attr="Google"
                    )
                    folium.raster_layers.ImageOverlay(
                        image=ruta_overlay,
                        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                        opacity=1.0,
                        name=f"Zona cremada ({ultim['data']})"
                    ).add_to(mapa_incendi)
                    st_folium(mapa_incendi, height=400, use_container_width=True,
                              key="mapa_incendi_overlay", returned_objects=[])
                    st.caption(
                        f"🔴 Zona cremada detectada fins a la darrera passada útil ({ultim['data']}): "
                        f"{ultim['hectarees_cremades']:.1f} ha. Situada sobre el mapa real, no només com "
                        f"a imatge solta, perquè es vegi la forma i l'extensió geogràfica de l'incendi."
                    )
                    st.markdown("---")

            st.markdown("##### Detall dia a dia:")
            for i in resultats:
                if i.get('alerta_creixement', False):
                    st.error(
                        f"🔥 **CREIXEMENT DETECTAT (Data: {i['data']})** — +{i['hectarees_noves']:.1f} ha "
                        f"cremades noves des de l'última passada útil (acumulat: {i['hectarees_cremades']:.1f} ha). "
                        f"**Decisió a bord:** prioritat de baixada ALTA per aquesta imatge."
                    )

                marca_referencia = "  📌 Referència" if i.get('es_referencia', False) else ""
                titol = (f"📅 Data: {i['data']}  |  🔥 {i['hectarees_cremades']:.2f} ha (acumulat)"
                         f"  |  ☁️ Núvols: {i['perc_nuvols']:.1f}%{marca_referencia}")
                with st.expander(titol):

                    if i.get('es_referencia', False):
                        st.info(
                            "📌 Aquesta és la imatge de REFERÈNCIA (abans de l'incendi / inici de la sèrie): "
                            "0 ha per definició. Per això el panell \"Zona Cremada\" surt tot negre — és "
                            "el resultat esperat, no un error."
                        )

                    c1,c2,c3 = st.columns(3)
                    with c1:
                        ruta_rgb = os.path.join(ruta_carpeta, i['rgb_png'])
                        if os.path.exists(ruta_rgb):
                            st.image(ruta_rgb, caption="1. Vista Real (Satèl·lit)", width='stretch')
                    with c2:
                        ruta_cloud = os.path.join(ruta_carpeta, i['cloud_png'])
                        if os.path.exists(ruta_cloud):
                            st.image(ruta_cloud, caption=f"2. Detecció de Núvols (AI Mask: {i['perc_nuvols']:.1f}%)", width='stretch')
                    with c3:
                        ruta_png_real = os.path.join(ruta_carpeta, i['imatge_png'])
                        if os.path.exists(ruta_png_real):
                            st.image(ruta_png_real, caption="3. Zona Cremada (dNBR)", width='stretch')

                    st.markdown("---")
                    col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
                    with col_metrica1:
                        st.metric(label="☁️ Cobertura de Núvols (IA):", value=f"{i['perc_nuvols']:.2f} %")
                    with col_metrica2:
                        st.metric(label="🔥 Cremat (acumulat):", value=f"{i['hectarees_cremades']:.2f} ha")
                    with col_metrica3:
                        st.metric(label="📈 Noves des de l'última:", value=f"{i['hectarees_noves']:.2f} ha")
                    st.markdown("---")

                    st.write(f"**Nom original: ** '{i['arxiu']}'  ({i['mida_bytes']/1024/1024:.1f} MB)")
                    st.write(f"**Estat:** Processat correctament a la GPU.")

                    ruta_imatge_real = os.path.join(ruta_carpeta, i['arxiu'])
                    if os.path.exists(ruta_imatge_real):
                        with open(ruta_imatge_real, "rb") as file:
                            st.download_button(
                                label="📥 Descarregar Imatge Real de Satèl·lit (.tif)",
                                data=file,
                                file_name=i['arxiu'],
                                mime="image/tiff",
                                key=i['arxiu']
                            )
                        st.info("💡 Nota: Els arxius .tif multispectrals requereixen programari GIS (com QGIS) per a la seva visualització.")

        with col_drt:
            st.subheader("📈 Evolució del perímetre cremat: ")
            analisis.generar_grafic_evolucio_incendi(resultats)

            # --- Panell explicatiu: per què importen la GPU i el temps aquí ---
            bytes_totals = st.session_state.get('bytes_totals_resultats') or 0
            n_utils = len(resultats)
            n_alertes = sum(1 for r in resultats if r.get('alerta_creixement', False))
            mb_totals = bytes_totals / 1024 / 1024
            # 1 KB per alerta: coordenades + hectàrees + percentatge, no la imatge sencera
            kb_alertes = n_alertes * 1
            if mb_totals > 0:
                estalvi_percentual = 100 * (1 - (kb_alertes/1024) / mb_totals) if n_alertes else 100.0
            else:
                estalvi_percentual = 0.0
            st.markdown("---")
            st.markdown("##### ⚡ Per què importen aquí la GPU i el temps de processament")
            st.info(
                f"S'han processat **{n_utils} imatges útils** (**{mb_totals:.1f} MB** en total) i se n'han "
                f"marcat **{n_alertes}** amb creixement significatiu (prioritat de baixada alta).\n\n"
                f"Si cada alerta es transmet com a missatge curt (coordenades + hectàrees, ~1 KB) en lloc "
                f"d'esperar a baixar la imatge sencera, l'estalvi il·lustratiu de dades per a la decisió "
                f"de prioritat és d'un **{estalvi_percentual:.1f}%**. La imatge original sempre queda "
                f"disponible per baixar-la sencera després, amb calma — el que es guanya en velocitat és "
                f"NOMÉS en la decisió de què cal prioritzar ara mateix.\n\n"
                f"⚠️ Aquesta comparació és il·lustrativa (mida real dels fitxers processats, però un "
                f"enllaç de baixada satèl·lit-terra concret dependria de la missió); no és l'especificació "
                f"d'un satèl·lit real."
            )

            # --- Gràfic: processar a bord (mesurat) vs. baixar-ho tot a Terra i processar-ho allà (estimat) ---
            AMPLADA_BANDA_ASSUMIDA_MBPS = 1.0  # enllaç modest en banda S, típic d'un nanosatèl·lit real
            temps_gpu_disp = st.session_state.get('temps_gpu_resultats') or 0
            temps_cpu_disp = st.session_state.get('temps_cpu_resultats') or 0
            temps_bord = temps_gpu_disp + temps_cpu_disp
            temps_baixada = (mb_totals * 8) / AMPLADA_BANDA_ASSUMIDA_MBPS  # MB -> Mbit / Mbps = segons
            st.markdown("##### 🛰️ A bord vs. baixar-ho tot a Terra primer")
            analisis.generar_grafic_comparacio_temps(temps_bord, temps_baixada, temps_bord, AMPLADA_BANDA_ASSUMIDA_MBPS)
            st.caption(
                f"**Barra verda (real, mesurada):** temps de GPU+CPU d'aquesta execució ({temps_bord:.1f} s). "
                f"**Barra vermella (estimada):** temps de baixar els {mb_totals:.1f} MB reals a un enllaç "
                f"il·lustratiu d'{AMPLADA_BANDA_ASSUMIDA_MBPS:g} Mbps ({temps_baixada:.1f} s), més el mateix "
                f"temps de processament un cop arribades a Terra (assumint maquinari equivalent). L'amplada "
                f"de banda és una suposició raonable per a un nanosatèl·lit, no l'especificació d'una missió."
            )

            st.write("---")
            st.subheader(" 🎥 Timelapse de l'incendi: ")
            ruta_gif_final = os.path.join(ruta_carpeta, "timelapse.gif")
            clau_gif = tuple(r['arxiu'] for r in resultats)
            if st.session_state.get('clau_gif') != clau_gif or not os.path.exists(ruta_gif_final):
                with st.spinner("Building wildfire timelapse over time..."):
                    # Vista real (RGB), no la màscara binària de cremat: l'incendi sol ser petit
                    # respecte al requadre i una màscara blanc/negre surt gairebé tota negra.
                    exit_gif = analisis.generar_timelapse(resultats, ruta_carpeta, ruta_gif_final, clau_imatge='rgb_png')
                st.session_state['clau_gif'] = clau_gif
                st.session_state['exit_gif'] = exit_gif
            else:
                exit_gif = st.session_state.get('exit_gif', False)
            if exit_gif:
                st.image(ruta_gif_final, width='stretch')
            else:
                st.warning("Timelapse couldn't be generated.")

    # ======================================================================================
    # MODE AIGUA (embassaments): prova de concepte de processament amb GPU a l'espai
    # ======================================================================================
    else:
        with col_esq:
            st.subheader("📊 Registre d'Observacions:")

            for i in resultats:
                #DIBUIXEM UN DESPLEGABLE PER CADA IMATGE, amb el percentatge de núvols al títol
                titol_avis = "  ⚠️ Resultat sospitós" if i.get('avis_boira', False) else ""
                with st.expander(f"📅 Data: {i['data']}  |  💧 {i['hectarees']:.2f} ha |  ☁️ Núvols: {i['perc_nuvols']:.1f}%{titol_avis}"):

                    # --- AVÍS: possible boira/cirrus que la IA de núvols no ha detectat ---
                    # (la IA està entrenada amb núvols opacs; la boira prima li passa desapercebuda però
                    # esborra el contrast que fa servir el NDWI per detectar l'aigua)
                    if i.get('avis_boira', False):
                        st.warning(
                            f"⚠️ **Resultat poc fiable.** La IA diu que aquesta imatge està neta "
                            f"({i['perc_nuvols']:.1f}% de núvols), però només detecta **{i['hectarees']:.2f} ha** "
                            f"d'aigua, molt per sota de la resta d'imatges clares d'aquesta sèrie "
                            f"(~{i['mediana_referencia']:.1f} ha). Podria haver-hi boira o cirrus que la IA no "
                            f"ha sabut detectar i que ha esborrat part de l'aigua del càlcul NDWI."
                        )

                    c1,c2,c3 = st.columns(3)

                    with c1:
                        ruta_rgb = os.path.join(ruta_carpeta, i['rgb_png'])
                        if os.path.exists(ruta_rgb):
                            st.image(ruta_rgb, caption="1. Vista Real (Satèl·lit)", width='stretch')

                    with c2:
                        ruta_cloud = os.path.join(ruta_carpeta, i['cloud_png'])
                        if os.path.exists(ruta_cloud):
                            st.image(ruta_cloud, caption=f"2. Detecció de Núvols (AI Mask: {i['perc_nuvols']:.1f}%)", width='stretch')

                    with c3:
                        ruta_png_real = os.path.join(ruta_carpeta, i['imatge_png'])
                        if os.path.exists(ruta_png_real):
                            st.image(ruta_png_real, caption="3. Detecció d'Aigua (NDWI)", width='stretch')

                    st.markdown("---")
                    col_metrica1, col_metrica2 = st.columns(2)
                    with col_metrica1:
                        st.metric(label="☁️ Cobertura de Núvols (Detectat amb IA):", value=f"{i['perc_nuvols']:.2f} %")
                    with col_metrica2:
                        st.metric(label="💧 Superfície d'Aigua:", value=f"{i['hectarees']:.2f} ha")
                    st.markdown("---")

                    st.write(f"**Nom original: ** '{i['arxiu']}'")
                    st.write(f"**Estat:** Processat correctament a la GPU.")

                    ruta_imatge_real = os.path.join(ruta_carpeta, i['arxiu'])
                    if os.path.exists(ruta_imatge_real):
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
            analisis.generar_grafic_evolucio(resultats)

            st.write("---")
            st.subheader(" 🎥 Timelapse Satel·lital: ")
            ruta_gif_final = os.path.join(ruta_carpeta, "timelapse.gif")

            # El GIF només es regenera si han canviat els resultats (i no a cada rerun de Streamlit,
            # p. ex. en clicar un botó de descàrrega)
            clau_gif = tuple(r['arxiu'] for r in resultats)
            if st.session_state.get('clau_gif') != clau_gif or not os.path.exists(ruta_gif_final):
                with st.spinner("Building satellite timelapse over time..."):
                    exit_gif = analisis.generar_timelapse(resultats, ruta_carpeta, ruta_gif_final)
                st.session_state['clau_gif'] = clau_gif
                st.session_state['exit_gif'] = exit_gif
            else:
                exit_gif = st.session_state.get('exit_gif', False)

            if exit_gif:
                st.image(ruta_gif_final, width='stretch')
            else:
                st.warning("Timelapse couldn't be generated.")
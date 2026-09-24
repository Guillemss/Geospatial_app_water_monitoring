
import os
import rasterio #per poder obrir imatges satelitals
import rasterio.warp #per convertir límits UTM a lat/lon (overlay del mapa en mode incendis)
import numpy as np #per poder utilitzar la GPU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
import torch
#posar import cupy as cp si s'utilitza un ordinador sense targeta gràfica NVIDIA
#!!!! modificar si tinc NVIDIA
import matplotlib.pyplot as plt #llibreria de gràfics que ens permetrà generar la imatge
import sys
import time
import subprocess # Necessari per parlar directament amb els sensors del BSC
import streamlit as st

# --- Paràmetres de l'avís de possible boira/cirrus no detectat (vegeu processar_directori) ---
LLINDAR_NUVOLS_REFERENCIA = 5.0  # % de núvols (segons la IA) per considerar una imatge "de referència"
FACTOR_SOSPITOS_BOIRA = 0.5      # si l'aigua és menys de la meitat de la mediana de referència, sospitem
MINIM_REFERENCIES_BOIRA = 2      # calen com a mínim 2 imatges clares a la sèrie per poder comparar


#--------------------CONNEXIÓ AMB IA de CLOUD DETECTION-----------------
# --- 1. AFEGIR EL RADAR PER LA IA DEL JANNIS ---
ruta_ai_model = os.path.join(os.path.dirname(__file__), "obpmark_ml_main")
if ruta_ai_model not in sys.path:
    sys.path.append(ruta_ai_model)

from src.semantic_segmentation.python.inference.backend_pytorch_native import BackendPytorchNative
#from obpmark_ml_main.src.semantic_segmentation.python.inference.backend_pytorch_native import BackendPytorchNative #importem aquesta classr que és la que té el motor de la IA.

#--------------------FUNCIÓ D'AI CLOUD DETECTION-----------------------
def reflectancia_a_entrada_ia(banda):
    # La xarxa es va entrenar amb Landsat 8 L1 (DN de 16 bits, normalitzat dividint entre 65535).
    # Sentinel-2 SR és reflectància x10000. Landsat: rho = 2e-5*DN - 0.1  =>  DN/65535 = (rho + 0.1)/1.31072
    # Multipliquem per 255 perquè model_ia.preprocess() ho torna a dividir entre 255 (queda a [0,1]).
    return np.clip((banda / 10000.0 + 0.1) / 1.31072, 0.0, 1.0) * 255.0


def ai_cloud_detection(banda_b,banda_verda, banda_r, banda_nir, model_ia):
    #Aquesta funció junta les capes i busca núvols amb la GPU

    # Apilem les 4 capes juntes en l'ordre del training: Vermell, Verd, Blau, NIR (R,G,B,NIR)
    imatge_4c = np.dstack([reflectancia_a_entrada_ia(b) for b in (banda_r, banda_verda, banda_b, banda_nir)]) #a funció np.dstack de NumPy permet posar les capes una sobre l'altra

    #Retallem les vores perque siguin múltiples de 32
    alçada, amplada = imatge_4c.shape[:2]
    nova_alcada = (alçada // 32) * 32
    nova_amplada = (amplada // 32) * 32
    imatge_4c = imatge_4c[:nova_alcada, :nova_amplada, :]

    #Executar el model de IA:
    # Convertim la imatge 3D en una "caixa" 4D (Batch de 1)
    feed = model_ia.preprocess(imatge_4c)
    if isinstance(feed, np.ndarray):
        feed = np.expand_dims(feed, axis=0) # Si és un array de NumPy
    else:
        feed = feed.unsqueeze(0) # Si ja ho ha convertit a tensor de PyTorch

    # ---GPU: Enviar les dades a la VRAM en el format correcte (float32) 
    #Assegurar-nos que és un tensor de PyTorch
    if not torch.is_tensor(feed):
        feed = torch.tensor(feed)
        
    #Enviar-lo a la VRAM de la gràfica i en format float32
    if torch.cuda.is_available():
        feed = feed.to('cuda', dtype=torch.float32)
    else:
        feed = feed.to(dtype=torch.float32)
    # ------------------------------------------------

    #Predicció
    pred = model_ia.predict(feed)
    
    # --- LA SOLUCIÓ: Extreure només el resultat principal ---
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    elif isinstance(pred, dict):
        pred = pred.get('out', list(pred.values())[0])

    #3 Postprocessar
    mascara_nuvols = model_ia.postprocess(pred)


    #Desempaquetar el resultat final
    if isinstance(mascara_nuvols, (list, tuple)):
        mascara_nuvols = mascara_nuvols[0] # Obrim la capsa i agafem la imatge
        
    # Assegurem-nos que és un array de NumPy (per poder fer .size)
    if hasattr(mascara_nuvols, 'cpu'):
        mascara_nuvols = mascara_nuvols.cpu().numpy()

    #Calcular quin perentatge de la imatge són núvols
    total_pixels = mascara_nuvols.size
    pixels_nuvol = np.count_nonzero(mascara_nuvols)
    percentatge_nuvols = (pixels_nuvol /total_pixels) *100

    #retornem el número del percentatge i la màscara de núvols
    return percentatge_nuvols, mascara_nuvols



#--------------------PROCESSAR IMATGE-----------------
def processar_imatge_aigua(ruta_imatge_tif, model_ia, limit_nuvols = 10):
    #Es calcula la superfície d'aigua d'una imatge amb la GPU
    #Retorna la màscara de 0 i 1 i el número d'hectàrees.
    #reb l'enllaç d'una sola imatge

    #1. Llegir la imatge a la CPU
    with rasterio.open(ruta_imatge_tif) as src:
        # L'ordre de descàrrega a GEE va ser: ['B2', 'B3', 'B4', 'B8']
        # Per tant, B3 (Verd) és la capa 2, i B8 (NIR) és la capa 4.
        banda_verda = src.read(2).astype('float32') #banda verda
        banda_nir = src.read(4).astype('float32') #banda Infraroig (NIR)
        banda_b = src.read(1).astype('float32')  # Blau
        banda_r = src.read(3).astype('float32')  # Vermell
        # Cal llegir les 4 bandes perquè funcioni la IA de detecció de núvols
        percentatge_nuvols, mascara_nuvols = ai_cloud_detection(banda_b,banda_verda, banda_r,banda_nir, model_ia)

        #Si la imatge està massa tapada, descartem l'operació
        if percentatge_nuvols > limit_nuvols:
            return None, None, percentatge_nuvols, None, None


        #Obtenir l'àrea d'un pixel: resolució 10x10 = 100m2
        transformacio = src.transform
        area_pixel_m2 = transformacio[0] * -transformacio[4] 
        #transformacio[0] ens diu quina és l'amplada d'un pixel i transformacio[4] ens diu alçada pixel
        #porta un signe negatiu pq a l'ordinador l'eix de les y creix cap avall pero
        # als mapes geògrafs l'eix de les Y creix cap amunt. La coordendada és negativa per indicar que ens estem movent cap amunt en el món real
        #les àrees no poden ser negatives

    #2. Enviar a la GPU (Simulació de l'Edge Computing al satèl·lit)
    #AQUÍ APLIQUEM LA PROGRAMACIÓ PARAL·LELA AMB CUDA
    # 2. Enviar a la GPU usant PyTorch (que ja té la NVIDIA configurada)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU si n'hi ha; si no, CPU (mateix criteri que la IA de núvols)
    gpu_verda = torch.tensor(banda_verda, device=device)
    gpu_nir = torch.tensor(banda_nir, device=device)

    #3. CÀLCUL MATEMÀTIC PARAL·LEL A LA GPU (NDWI)
    #Fòrmula: NDWI = (Verd-NIR) /(Verd+NIR)
    # 3. CÀLCUL MATEMÀTIC PARAL·LEL A LA GPU (NDWI)
    denominador = (gpu_verda + gpu_nir)
    denominador[denominador == 0] = 0.0001

    ndwi_gpu = (gpu_verda - gpu_nir) / denominador
#
    #NDWI > 0 és aigua
    mascara_aigua_gpu = ndwi_gpu > 0.15 #Si augmentem aquest valor, es fa més estricte i no es pensa que les ombres de núvols i muntanyes és aigua


    #4. EXTRACCIÓ DEL RESULTAT (Tornar només allò important a la Terra)
    #total_pixels_aigua = np.sum(mascara_aigua_gpu)
    # Comptem els píxels directament a la GPU usant CuPy
    # 4. EXTRACCIÓ DEL RESULTAT 
    # Comptem els píxels a la GPU i ho passem a número normal amb .item()
    total_pixels_aigua = torch.count_nonzero(mascara_aigua_gpu).item()
    hectarees = float((total_pixels_aigua * area_pixel_m2) / 10000.0)

    # Fabriquem la imatge RGB visual
    rgb = np.dstack((banda_r, banda_verda, banda_b))
    img_rgb = np.clip(rgb / 3000.0, 0, 1) # Ajust de brillantor pel satèl·lit

    # Convertim la màscara de la GPU un altre cop cap a la CPU per dibuixar-la
    mascara_aigua_cpu = mascara_aigua_gpu.cpu().numpy()

    return mascara_aigua_cpu, hectarees, percentatge_nuvols, mascara_nuvols, img_rgb


#--------------------PROCESSAMENT DE LES IMATGES-----------------
def processar_directori(carpeta_imatges):
    #Llegeix totes es imatges d'una carpeta, les processa a la GPU i retorna una llista amb l'evolució de l'aigua al llarg del temps.

    resultats = []
    print("\n> INICIANT CONNEXIÓ AMB GPU AL BSC...")
    print("Iniciant processament amb la GPU de la carpeta: " + str(carpeta_imatges))

    model_ia = BackendPytorchNative()
    ruta_pth = os.path.join(os.path.dirname(__file__), 'obpmark_ml_main', 'src', 'semantic_segmentation', 'models', 'pytorch', 'fp32', 'state_dict.pth')
    model_ia.load(model_path = ruta_pth)


    #busquem tots els arxius .tif de la carpeta
    arxius = [f for f in os.listdir(carpeta_imatges) if f.endswith('.tif')]
    total_fotos = len(arxius) #Quantes fotos ha descarregat el satèl·lit


    #Per poder calcular la velocitat de la GPU
    temps_total_gpu = 0 #per guardar el temps total de la GPU
    temps_total_cpu = 0

    for index, arxiu in enumerate(arxius, start =1):
        ruta_completa = os.path.join(carpeta_imatges, arxiu)

        #Per cada imatge cridem a la funció principal
        inici_gpu = time.time()#comença el cronometre de la GPU (IA + NDWI)
        mascara, hectarees, perc_nuvols, mascara_nuvols, img_rgb = processar_imatge_aigua(ruta_completa, model_ia)
        temps_gpu = time.time()- inici_gpu
        temps_total_gpu += temps_gpu

        print(f"> [IA ACTIVA] Analtzant Foto {index}/{total_fotos} ({arxiu})...")
        print(f"  Temps GPU: {temps_gpu:.3f} segons")        
        print(f"  ☁️ S'ha detectat un {perc_nuvols:.2f}% de núvols.")

        #st.toast(f"☁️ Analitzant {arxiu}: S'ha detectat un {perc_nuvols:.2f}% de núvols.")

        # Enviem l'avís a la terminal de la web (Ara ho fem amb print gràcies al Hack de sys.stdout!)

        # Simulem el satèl·lit: si la màscara és None, ho esborrem
        if mascara is None:
            print(f"  ❌ DESCARTADA: Imatge {arxiu} rebutjada al satèl·lit (Massa núvols: {perc_nuvols:.1f}%)\n")
            os.remove(ruta_completa)
            continue
            
        print(f"  ✅ OK: Imatge vàlida | Aigua detectada: {hectarees:.2f} ha\n")

        inici_cpu = time.time() #Iniciem el cronometre per la CPU(guardar imatges i resultats)

        # vmin/vmax fixos perquè una màscara tota a True no es pinti com si fos tota a False
        nom_png = arxiu.replace('.tif', '_mask.png')
        ruta_png = os.path.join(carpeta_imatges, nom_png)
        plt.imsave(ruta_png, mascara, cmap = 'Blues', vmin=0, vmax=1)

        # Guardar imatge de NÚVOLS de la IA (Blanc i negre)
        nom_cloud = arxiu.replace('.tif', '_cloud.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_cloud), mascara_nuvols, cmap='gray', vmin=0, vmax=1)

        #  Guardar imatge REAL RGB (Color real)
        nom_rgb = arxiu.replace('.tif', '_rgb.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_rgb), img_rgb)

        temps_cpu = time.time()-inici_cpu
        temps_total_cpu += temps_cpu
        print(f"  Temps CPU (Guardar gràfics): {temps_cpu:.3f} segons\n")

        #agafem la data en la que sha fet la foto de la imatge (els 8 primers caràcters del nom)
        data_crua = arxiu[:8]
        data_neta = str(data_crua[6:8])+"/"+str(data_crua[4:6])+"/"+str(data_crua[:4]) # ho passem a format '08/04/2026

        #Guardem el nomde l'arxiu i de les hectàrees calcualdes
        #és un diccionari que l'afegim a una llista
        # Guardem el nom de l'arxiu, les hectàrees, els núvols i totes les fotos
        dic = {
            'arxiu': arxiu,
            'imatge_png': nom_png, 
            'cloud_png': nom_cloud, 
            'rgb_png': nom_rgb, 
            'data': data_neta, 
            'hectarees': hectarees,
            'perc_nuvols': perc_nuvols,
            'avis_boira': False, # es marca més avall si el resultat sembla poc fiable
            'mediana_referencia': None
        }
        resultats.append(dic)

    #Ordenem la lista de diccionaris per la DATA en la que sha fet la foto! --> ens fixem en el nom de l'arxiu, que sempra comença per YYYYMMDD
    resultats = sorted(resultats, key = lambda x: x['arxiu'])

    # --- AVÍS DE POSSIBLE BOIRA/CIRRUS NO DETECTAT PER LA IA ---
    # La IA de núvols està entrenada amb núvols opacs (dataset 38-Cloud, Landsat-8) i no veu bé la boira
    # prima o el cirrus d'alçada: quan n'hi ha, diu que la imatge està neta (% de núvols baix) però el
    # NDWI surt molt per sota del real, perquè la boira "esborra" el contrast verd/NIR que delata l'aigua.
    # No depenem de cap dada externa: la MEDIANA es calcula només amb les imatges MOLT clares
    # (referència de confiança), però la comprovació es fa sobre TOTES les imatges de resultats (que ja
    # han passat el filtre principal de la IA, limit_nuvols). Si només comprovéssim les molt clares,
    # una imatge com "5.3% de núvols, però només 2 ha d'aigua" quedaria fora de la comprovació just per
    # estar lleugerament per sobre del llindar de referència, i és justament el cas que volem detectar.
    referencies = [r['hectarees'] for r in resultats if r['perc_nuvols'] <= LLINDAR_NUVOLS_REFERENCIA]
    if len(referencies) >= MINIM_REFERENCIES_BOIRA:
        referencies.sort()
        mediana = referencies[len(referencies)//2]
        if mediana > 0: # si la mediana és 0 (p.ex. zona sense aigua) no té sentit comparar percentatges
            for r in resultats:
                if r['hectarees'] < mediana * FACTOR_SOSPITOS_BOIRA:
                    r['avis_boira'] = True
                    r['mediana_referencia'] = round(mediana, 1)
                    print(f"  ⚠️ AVÍS: {r['arxiu']} dona {r['hectarees']:.1f} ha (IA diu {r['perc_nuvols']:.1f}% núvols) "
                          f"però la resta d'imatges clares d'aquesta sèrie donen ~{mediana:.1f} ha. "
                          f"Possible boira/cirrus no detectat: resultat poc fiable.")

    print("> PROCÉS COMPLETAT AMB ÈXIT!\n")
    print(f"TEMPS TOTAL GPU: {temps_total_gpu:.2f} s | TEMPS TOTAL CPU: {temps_total_cpu:.2f} s\n")

    return resultats, round(temps_total_gpu, 2), round(temps_total_cpu, 2)

#========================================================================================
#--------------------CAS D'ÚS 2: INCENDIS (resposta en temps crític)-----------------
#========================================================================================
# Diferència clau amb l'aigua: aquí SÍ importa la velocitat. Un embassament no canvia gaire
# d'una passada del satèl·lit a la següent (uns dies), així que processar-lo més ràpid o més
# lent amb GPU no canvia res de pràctic: és la prova de concepte que demostra que es PUOT fer
# processament amb GPU a l'espai. Un incendi, en canvi, es propaga en hores: si el satèl·lit
# no decideix a l'instant (a bord, amb GPU) quina imatge val la pena prioritzar per baixar,
# la informació arriba tard o es queda en cua darrere de dades menys urgents.
# Reaprofitem tota la infraestructura del cas de l'aigua: la mateixa IA de núvols
# (ai_cloud_detection) fa de "porter" (decideix si la imatge és utilitzable); la detecció de
# l'incendi en si NO fa servir cap xarxa neuronal, és matemàtica pura a la GPU (com el NDWI).

# Llindar estàndard a la literatura de teledetecció per considerar "cremat" un píxel amb dNBR
LLINDAR_DNBR_CREMAT = 0.25
# Creixement mínim (ha noves cremades respecte a l'última observació útil) per marcar l'alerta.
# Comprovat amb dades reals (Bisbal, abans que comencés l'incendi): el soroll normal de dNBR + petites
# imprecisions de la IA de núvols ja acumula 5-40 ha sense cap incendi real. Amb un llindar de 5 ha,
# una imatge d'ABANS de l'incendi (p.ex. 27/06, +5.9 ha) disparava una alerta de "creixement" que
# semblava un fals positiu. 50 ha queda per sobre d'aquest soroll i molt per sota d'un salt real
# (l'incendi de la Bisbal va fer un salt de +1.960 ha en una sola passada).
LLINDAR_CREIXEMENT_ALERTA = 50.0
# Marge (píxels) que s'exclou al voltant de cada núvol detectat abans de buscar zona cremada.
# Els píxels a la VORA d'un núvol no són ni núvol net ni terreny net (llum difusa, mig tapats):
# el seu dNBR pot sortir molt alt sense haver-hi cap incendi, i el resultat sembla "resseguir" la
# forma del núvol. Comprovat amb dades reals: sense marge, una imatge amb un 12% de núvols donava
# ~209 ha de "cremat" fals; amb un marge de 15 píxels (150 m), baixa a ~15 ha (soroll normal).
MARGE_NUVOL_PX = 15
# Llindar de classificació per al MODE GRAELLA (escaneig), diferent de LLINDAR_CREIXEMENT_ALERTA.
# Aquell és per detectar un salt d'un dia a l'altre dins la sèrie detallada d'UNA zona ja coneguda.
# Aquí cal distingir "hi ha un incendi real en aquesta tessel·la" de "soroll normal de fons"
# (comprovat amb dades reals: una tessel·la sense cap incendi acumulava 10-40 ha de soroll residual,
# ~0,1-0,2% de l'àrea, per soroll de dNBR i petites imprecisions de la IA de núvols). 100 ha és molt
# per sobre d'aquest soroll i molt per sota de l'incendi real detectat (~2.000 ha).
LLINDAR_GRAELLA_HA = 100.0


def dilatar_mascara(mascara_bool, marge_px):
    #Eixampla (dilata) una màscara booleana 'marge_px' píxels en totes direccions.
    #Fem servir un max-pool perquè ja tenim torch com a dependència (no cal afegir scipy).
    if marge_px <= 0:
        return mascara_bool
    tensor = mascara_bool.to(torch.float32)[None, None]
    mida_nucli = 2 * marge_px + 1
    dilatat = torch.nn.functional.max_pool2d(tensor, kernel_size=mida_nucli, stride=1, padding=marge_px)
    return dilatat[0, 0] > 0.5


def calcular_nbr(banda_nir, banda_swir2, device):
    #NBR (Normalized Burn Ratio) = (NIR - SWIR2) / (NIR + SWIR2)
    #Vegetació sana: NBR alt. Vegetació cremada: NBR baix (el SWIR puja perquè ja no hi ha fulles/aigua).
    gpu_nir = torch.tensor(banda_nir, device=device)
    gpu_swir2 = torch.tensor(banda_swir2, device=device)
    denominador = gpu_nir + gpu_swir2

    # Píxels sense dades (vora del mosaic/mostreig: totes les bandes a 0) tenen denominador 0.
    # NBR hi donaria 0 (vegetació sana sol tenir NBR positiu), i una resta baseline−0 sembla "cremat"
    # encara que només sigui una vora sense dades. Marquem aquests píxels com a NO vàlids.
    mascara_valida = denominador != 0

    denominador_segur = denominador.clone()
    denominador_segur[~mascara_valida] = 0.0001
    nbr = (gpu_nir - gpu_swir2) / denominador_segur
    return nbr, mascara_valida


def processar_imatge_incendi(ruta_imatge_tif, model_ia, limit_nuvols = 10):
    #Igual que processar_imatge_aigua, però calcula el NBR en lloc del NDWI.
    #Retorna el tensor NBR (a la GPU, encara sense comparar amb cap referència), no pas hectàrees:
    #cal una imatge "abans de l'incendi" per saber què ha canviat (vegeu processar_directori_incendi).

    with rasterio.open(ruta_imatge_tif) as src:
        # Ordre de descàrrega (BANDES_FOC a data_extraction_2.py): B2,B3,B4,B8,B11,B12
        banda_b = src.read(1).astype('float32')
        banda_verda = src.read(2).astype('float32')
        banda_r = src.read(3).astype('float32')
        banda_nir = src.read(4).astype('float32')
        banda_swir2 = src.read(6).astype('float32')  # B12

        # Mateixa IA de núvols que l'aigua: només necessita B,G,R,NIR
        percentatge_nuvols, mascara_nuvols = ai_cloud_detection(banda_b, banda_verda, banda_r, banda_nir, model_ia)

        if percentatge_nuvols > limit_nuvols:
            return None, percentatge_nuvols, None, None, None, None, None, None

        transformacio = src.transform
        area_pixel_m2 = transformacio[0] * -transformacio[4]
        crs_origen = src.crs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nbr_gpu, mascara_valida_gpu = calcular_nbr(banda_nir, banda_swir2, device)

    # El NBR és un índex per a VEGETACIÓ: sobre aigua (mar, embassaments, rius) les reflectàncies de
    # NIR i SWIR són molt baixes i sorolloses, i petites variacions es converteixen en canvis grans de
    # la ràtio, que es confonen amb "cremat" (comprovat amb dades reals: la costa marina sortia com a
    # zona cremada). Calculem un NDWI (el mateix índex del mode aigua) per marcar aquests píxels i
    # excloure'ls sempre de la detecció d'incendi.
    gpu_verda = torch.tensor(banda_verda, device=device)
    gpu_nir_aigua = torch.tensor(banda_nir, device=device)
    denominador_aigua = gpu_verda + gpu_nir_aigua
    denominador_aigua[denominador_aigua == 0] = 0.0001
    mascara_aigua_gpu = ((gpu_verda - gpu_nir_aigua) / denominador_aigua) > 0.15

    # ai_cloud_detection() retalla la imatge a un múltiple de 32 (mascara_nuvols és més petita que la
    # imatge original); el dNBR ha de tenir la MATEIXA mida per poder-los combinar píxel a píxel.
    h, w = mascara_nuvols.shape
    nbr_gpu = nbr_gpu[:h, :w]
    mascara_valida_gpu = mascara_valida_gpu[:h, :w]
    mascara_aigua_gpu = mascara_aigua_gpu[:h, :w]

    # Imatge RGB per la vista real (mateix ajust de brillantor que l'aigua)
    rgb = np.dstack((banda_r, banda_verda, banda_b))[:h, :w]
    img_rgb = np.clip(rgb / 3000.0, 0, 1)

    # Els píxels sense dades (vora del mosaic; vegeu calcular_nbr) sortirien negres per pura
    # coincidència (0/3000=0), i es confonen amb una imatge trencada o tallada. Els pintem d'un gris
    # clarament diferent del terreny real, perquè es vegi que és una vora sense cobertura, no un error.
    mascara_valida_np = mascara_valida_gpu.cpu().numpy()
    img_rgb[~mascara_valida_np] = 0.55

    # Límits geogràfics (lat/lon) de la imatge retallada (h,w): calen per situar la màscara de cremat
    # sobre el mapa interactiu (folium.raster_layers.ImageOverlay). La imatge original és en UTM
    # (metres); folium necessita WGS84 (lat/lon).
    esquerra, baix, dreta, dalt = rasterio.transform.array_bounds(h, w, transformacio)
    lon_min, lat_min, lon_max, lat_max = rasterio.warp.transform_bounds(crs_origen, 'EPSG:4326', esquerra, baix, dreta, dalt)
    limits_geo = (lat_min, lon_min, lat_max, lon_max)

    return nbr_gpu, percentatge_nuvols, mascara_nuvols, img_rgb, area_pixel_m2, mascara_valida_gpu, mascara_aigua_gpu, limits_geo


def processar_directori_incendi(carpeta_imatges, llindar_creixement_alerta = LLINDAR_CREIXEMENT_ALERTA):
    #Processa totes les imatges d'una carpeta i calcula la superfície CREMADA respecte a la
    #primera imatge útil de la sèrie (la "referència", d'abans de l'incendi).
    #Retorna la mateixa forma que processar_directori (resultats, temps_gpu, temps_cpu) més els
    #bytes totals de les imatges .tif originals, per poder explicar l'estalvi d'ample de banda.

    resultats = []
    print("\n> INICIANT CONNEXIÓ AMB GPU AL BSC (MODE INCENDIS)...")
    print("Iniciant processament amb la GPU de la carpeta: " + str(carpeta_imatges))

    model_ia = BackendPytorchNative()
    ruta_pth = os.path.join(os.path.dirname(__file__), 'obpmark_ml_main', 'src', 'semantic_segmentation', 'models', 'pytorch', 'fp32', 'state_dict.pth')
    model_ia.load(model_path = ruta_pth)

    arxius = sorted(f for f in os.listdir(carpeta_imatges) if f.endswith('.tif')) #ordenats per data (comencen per YYYYMMDD)
    total_fotos = len(arxius)

    temps_total_gpu = 0
    temps_total_cpu = 0
    bytes_totals_imatges = 0

    nbr_referencia = None       # NBR de la primera imatge útil (abans de l'incendi, o l'inici de la sèrie)
    mascara_valida_referencia = None
    mascara_aigua_referencia = None  # mar/embassaments/rius: mai poden ser "cremat" (vegeu processar_imatge_incendi)
    arxiu_referencia = None
    hectarees_acumulades_anterior = 0.0  # per calcular "quantes ha NOVES des de l'última observació"

    for index, arxiu in enumerate(arxius, start = 1):
        ruta_completa = os.path.join(carpeta_imatges, arxiu)
        mida_bytes = os.path.getsize(ruta_completa) #pes real de la imatge crua, abans de tocar-la

        inici_gpu = time.time()
        nbr_gpu, perc_nuvols, mascara_nuvols, img_rgb, area_pixel_m2, mascara_valida_gpu, mascara_aigua_gpu, limits_geo = processar_imatge_incendi(ruta_completa, model_ia)
        temps_gpu = time.time() - inici_gpu
        temps_total_gpu += temps_gpu

        print(f"> [IA ACTIVA] Analitzant Foto {index}/{total_fotos} ({arxiu})...")
        print(f"  Temps GPU: {temps_gpu:.3f} segons")
        print(f"  ☁️ S'ha detectat un {perc_nuvols:.2f}% de núvols.")

        if nbr_gpu is None:
            print(f"  ❌ DESCARTADA: Imatge {arxiu} rebutjada al satèl·lit (Massa núvols: {perc_nuvols:.1f}%)\n")
            os.remove(ruta_completa)
            continue

        bytes_totals_imatges += mida_bytes
        inici_cpu = time.time()

        # --- LA PRIMERA IMATGE ÚTIL ES CONVERTEIX EN REFERÈNCIA (abans de l'incendi) ---
        if nbr_referencia is None:
            nbr_referencia = nbr_gpu
            mascara_valida_referencia = mascara_valida_gpu
            mascara_aigua_referencia = mascara_aigua_gpu
            arxiu_referencia = arxiu
            mascara_cremat = torch.zeros_like(nbr_gpu, dtype=torch.bool) #per definició, 0 ha cremades a la referència
            print(f"  📌 Imatge de REFERÈNCIA (abans de l'incendi / inici de la sèrie).\n")
        else:
            dnbr = nbr_referencia - nbr_gpu #positiu = ha baixat el NBR = possible zona cremada
            mascara_nuvols_bool = torch.tensor(mascara_nuvols.astype(bool), device=dnbr.device)
            # Eixamplem la màscara de núvols abans d'excloure-la: la VORA d'un núvol té un dNBR fals
            # (ni núvol net ni terreny net) que, sense aquest marge, es confon amb "cremat" resseguint
            # la forma del núvol (vegeu MARGE_NUVOL_PX).
            mascara_nuvols_eixamplada = dilatar_mascara(mascara_nuvols_bool, MARGE_NUVOL_PX)
            # Excloem núvols (+ vora), píxels sense dades vàlides (referència o actual), I aigua (mar,
            # embassaments, rius: el NBR no és fiable sobre aigua, vegeu processar_imatge_incendi).
            mascara_cremat = ((dnbr > LLINDAR_DNBR_CREMAT) & ~mascara_nuvols_eixamplada
                              & mascara_valida_referencia & mascara_valida_gpu
                              & ~mascara_aigua_referencia & ~mascara_aigua_gpu)

        total_pixels_cremats = torch.count_nonzero(mascara_cremat).item()
        hectarees_cremades = float((total_pixels_cremats * area_pixel_m2) / 10000.0)
        hectarees_noves = max(0.0, hectarees_cremades - hectarees_acumulades_anterior)
        hectarees_acumulades_anterior = hectarees_cremades

        alerta_creixement = hectarees_noves > llindar_creixement_alerta
        if alerta_creixement:
            print(f"  🔥 [EDGE AI] CREIXEMENT DETECTAT: +{hectarees_noves:.1f} ha cremades des de l'última passada útil.")
            print(f"  📡 Prioritat de baixada ALTA per {arxiu} (canvi significatiu respecte a la referència).\n")

        print(f"  ✅ OK: Imatge vàlida | Zona cremada (acumulat): {hectarees_cremades:.2f} ha\n")

        mascara_cremat_cpu = mascara_cremat.cpu().numpy()

        # vmin/vmax fixos, igual que amb l'aigua (perquè una màscara tota a True/False no es pinti malament)
        nom_png = arxiu.replace('.tif', '_mask.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_png), mascara_cremat_cpu, cmap = 'hot', vmin=0, vmax=1)

        nom_cloud = arxiu.replace('.tif', '_cloud.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_cloud), mascara_nuvols, cmap='gray', vmin=0, vmax=1)

        nom_rgb = arxiu.replace('.tif', '_rgb.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_rgb), img_rgb)

        # Overlay transparent per situar la zona cremada SOBRE EL MAPA (no només com a imatge solta):
        # vermell allà on hi ha cremat, totalment transparent (alpha=0) a la resta, perquè es vegi la
        # forma real de l'incendi damunt del mapa de satèl·lit, no només en una miniatura RGB estàtica.
        nom_overlay = arxiu.replace('.tif', '_overlay.png')
        overlay_rgba = np.zeros((*mascara_cremat_cpu.shape, 4), dtype=np.float32)
        overlay_rgba[mascara_cremat_cpu, 0] = 1.0   # vermell
        overlay_rgba[mascara_cremat_cpu, 3] = 0.75  # opac només on hi ha cremat
        plt.imsave(os.path.join(carpeta_imatges, nom_overlay), overlay_rgba)

        temps_cpu = time.time() - inici_cpu
        temps_total_cpu += temps_cpu
        print(f"  Temps CPU (Guardar gràfics): {temps_cpu:.3f} segons\n")

        data_crua = arxiu[:8]
        data_neta = str(data_crua[6:8])+"/"+str(data_crua[4:6])+"/"+str(data_crua[:4])

        dic = {
            'arxiu': arxiu,
            'imatge_png': nom_png,
            'cloud_png': nom_cloud,
            'rgb_png': nom_rgb,
            'overlay_png': nom_overlay,
            'limits_geo': limits_geo, # (lat_min, lon_min, lat_max, lon_max) per situar l'overlay al mapa
            'data': data_neta,
            'hectarees_cremades': hectarees_cremades,
            'hectarees_noves': hectarees_noves,
            'perc_nuvols': perc_nuvols,
            'es_referencia': (arxiu == arxiu_referencia),
            'alerta_creixement': alerta_creixement,
            'mida_bytes': mida_bytes,
        }
        resultats.append(dic)

    resultats = sorted(resultats, key = lambda x: x['arxiu'])
    n_descartades = total_fotos - len(resultats) #imatges rebutjades per massa núvols (esborrades a bord)

    print("> PROCÉS COMPLETAT AMB ÈXIT!\n")
    print(f"TEMPS TOTAL GPU: {temps_total_gpu:.2f} s | TEMPS TOTAL CPU: {temps_total_cpu:.2f} s\n")
    print(f"DADES PROCESSADES: {bytes_totals_imatges/1024/1024:.1f} MB en {len(resultats)} imatges útils "
          f"({n_descartades} descartades per núvols).\n")

    return resultats, round(temps_total_gpu, 2), round(temps_total_cpu, 2), bytes_totals_imatges, n_descartades


def gpu_disponible():
    # True si PyTorch veu una GPU NVIDIA (CUDA). Si és False, tot s'executa en CPU.
    return torch.cuda.is_available()


def obtenir_estadistiques_hardware():
    # Funció per extreure el nom i la memòria de la targeta gràfica
    if torch.cuda.is_available():
        nom_gpu = torch.cuda.get_device_name(0)
        try:
            # Llegim el consum REAL dels sensors de la NVIDIA (comanda nvidia-smi)
            comanda = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
            vram_str = subprocess.check_output(comanda).decode('utf-8').strip()
            # Agafem la dada i la passem de MB a GB
            memoria_usada = round(float(vram_str.split('\n')[0]) / 1024, 2) 
        except Exception:
            # Si per algun motiu el sensor falla, usem la reserva global de PyTorch
            memoria_usada = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        
        return nom_gpu, memoria_usada
    else:
        return "CPU (Simulada)", 0
    

#Configurem que ha de fer el programa quan s'executi    
# Bloc de prova per executar l'arxiu de forma independent
if __name__ == "__main__":
    # Els emojis dels print no han de fer caure el programa en consoles que no són UTF-8 (Windows cp1252)
    sys.stdout.reconfigure(errors='replace')
    # Assegura't de posar la ruta correcta on l'escript anterior ha guardat les imatges
    carpeta_prova = os.path.join(os.path.expanduser('~'), 'Downloads', 'Prova_Sau_Sentinel2')
    
    if os.path.exists(carpeta_prova):
        dades = processar_directori(carpeta_prova)
        print("\nResum Final:")
        for dada in dades:
            print(f"{dada['arxiu']}: {dada['hectarees']:.2f} ha")
    else:
        print(f"No s'ha trobat la carpeta {carpeta_prova}. Revisa la ruta.")
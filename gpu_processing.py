
import os
import rasterio #per poder obrir imatges satelitals
import numpy as np #per poder utilitzar la GPU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
import torch
#posar import cupy as cp si s'utilitza un ordinador sense targeta gràfica NVIDIA
#!!!! modificar si tinc NVIDIA
import matplotlib.pyplot as plt #llibreria de gràfics que ens permetrà generar la imatge
import sys
import streamlit as st


#--------------------CONNEXIÓ AMB IA de CLOUD DETECTION-----------------
# --- 1. AFEGIR EL RADAR PER LA IA DEL JANNIS ---
ruta_ai_model = os.path.join(os.path.dirname(__file__), "obpmark_ml_main")
if ruta_ai_model not in sys.path:
    sys.path.append(ruta_ai_model)

from src.semantic_segmentation.python.inference.backend_pytorch_native import BackendPytorchNative
#from obpmark_ml_main.src.semantic_segmentation.python.inference.backend_pytorch_native import BackendPytorchNative #importem aquesta classr que és la que té el motor de la IA.

#--------------------FUNCIÓ D'AI CLOUD DETECTION-----------------------
def ai_cloud_detection(banda_b,banda_verda, banda_r, banda_nir, model_ia):
    #Aquesta funció junta les capes i busca núvols amb la GPU

    # Apilem les 4 capes juntes --> LA IA del Jannis mira fotos en color real
    imatge_4c = np.dstack((banda_b, banda_verda, banda_r, banda_nir)) #a funció np.dstack de NumPy permet posar les capes una sobre l'altra

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

    # --- PURIFICACIÓ DE DADES (El truc definitiu) ---
    if hasattr(feed, 'cpu'):
        feed = feed.cpu().numpy()
        
    # Obligem l'ordinador a usar 'float64' perquè PyTorch no s'espanti
    feed = np.array(feed).astype('float64')
    # ------------------------------------------------

    #Predicció
    pred = model_ia.predict(feed)
    
    # --- LA SOLUCIÓ: Extraiem només el resultat principal ---
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
    device = torch.device('cuda') # Forcem la targeta gràfica
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
    print("Iniciant processament amb la GPU de la carpeta: " + str(carpeta_imatges))

    model_ia = BackendPytorchNative()
    ruta_pth = os.path.join(os.path.dirname(__file__), 'obpmark_ml_main', 'src', 'semantic_segmentation', 'models', 'pytorch', 'fp32', 'state_dict.pth')
    model_ia.load(model_path = ruta_pth)

    #busquem tots els arxius .tif de la carpeta
    arxius = [f for f in os.listdir(carpeta_imatges) if f.endswith('.tif')]

    for arxiu in arxius:
        ruta_completa = os.path.join(carpeta_imatges, arxiu)

        #Per cada imatge cridem a la funció principal
        mascara, hectarees, perc_nuvols, mascara_nuvols, img_rgb = processar_imatge_aigua(ruta_completa, model_ia)
        print(f"☁️ Analitzant {arxiu}: S'ha detectat un {perc_nuvols:.2f}% de núvols.")

        #st.toast(f"☁️ Analitzant {arxiu}: S'ha detectat un {perc_nuvols:.2f}% de núvols.")


        # Simulem el satèl·lit: si la màscara és None, ho esborrem
        if mascara is None:
            print(f"❌ Imatge {arxiu} descartada al satèl·lit. Massa núvols: {perc_nuvols:.1f}%")
            os.remove(ruta_completa)
            continue

        nom_png = arxiu.replace('.tif', '_mask.png')
        ruta_png = os.path.join(carpeta_imatges, nom_png)

        plt.imsave(ruta_png, mascara, cmap = 'Blues')

        # 2. NOU: Guardar imatge de NÚVOLS de la IA (Blanc i negre)
        nom_cloud = arxiu.replace('.tif', '_cloud.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_cloud), mascara_nuvols, cmap='gray')
        
        # 3. NOU: Guardar imatge REAL RGB (Color real)
        nom_rgb = arxiu.replace('.tif', '_rgb.png')
        plt.imsave(os.path.join(carpeta_imatges, nom_rgb), img_rgb)

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
            'perc_nuvols': perc_nuvols
        }
        resultats.append(dic)

    #Ordenem la lista de diccionaris per la DATA en la que sha fet la foto! --> ens fixem en el nom de l'arxiu, que sempra comença per YYYYMMDD
    resultats = sorted(resultats, key = lambda x: x['arxiu'])

    return resultats
#Pero com es fa perque vagi passant el nom de cada imatge diferent???????????????????

#Configurem que ha de fer el programa quan s'executi    
# Bloc de prova per executar l'arxiu de forma independent
if __name__ == "__main__":
    # Assegura't de posar la ruta correcta on l'escript anterior ha guardat les imatges
    carpeta_prova = os.path.join(os.path.expanduser('~'), 'Downloads', 'Prova_Sau_Sentinel2')
    
    if os.path.exists(carpeta_prova):
        dades = processar_directori(carpeta_prova)
        print("\nResum Final:")
        for dada in dades:
            print(f"{dada['arxiu']}: {dada['hectarees']:.2f} ha")
    else:
        print(f"No s'ha trobat la carpeta {carpeta_prova}. Revisa la ruta.")
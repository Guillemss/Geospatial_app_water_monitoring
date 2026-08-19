
import os
import rasterio #per poder obrir imatges satelitals
import numpy as np #per poder utilitzar la GPU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
#posar import cupy as cp si s'utilitza un ordinador sense targeta gràfica NVIDIA
#!!!! modificar si tinc NVIDIA
import matplotlib.pyplot as plt #llibreria de gràfics que ens permetrà generar la imatge


#--------------------PROCESSAR IMATGE-----------------
def processar_imatge_aigua(ruta_imatge_tif):
    #Es calcula la superfície d'aigua d'una imatge amb la GPU
    #Retorna la màscara de 0 i 1 i el número d'hectàrees.
    #reb l'enllaç d'una sola imatge

    #1. Llegir la imatge a la CPU
    with rasterio.open(ruta_imatge_tif) as src:
        # L'ordre de descàrrega a GEE va ser: ['B2', 'B3', 'B4', 'B8']
        # Per tant, B3 (Verd) és la capa 2, i B8 (NIR) és la capa 4.
        banda_verda = src.read(2).astype('float32')
        banda_nir = src.read(4).astype('float32')

        #Obtenir l'àrea d'un pixel: resolució 10x10 = 100m2
        transformacio = src.transform
        area_pixel_m2 = transformacio[0] * -transformacio[4] 
        #transformacio[0] ens diu quina és l'amplada d'un pixel i transformacio[4] ens diu alçada pixel
        #porta un signe negatiu pq a l'ordinador l'eix de les y creix cap avall pero
        # als mapes geògrafs l'eix de les Y creix cap amunt. La coordendada és negativa per indicar que ens estem movent cap amunt en el món real
        #les àrees no poden ser negatives

    #2. Enviar a la GPU (Simulació de l'Edge Computing al satèl·lit)
    #AQUÍ APLIQUEM LA PROGRAMACIÓ PARAL·LELA AMB CUDA
    gpu_verda = np.array(banda_verda)
    gpu_nir = np.array(banda_nir)

    #3. CÀLCUL MATEMÀTIC PARAL·LEL A LA GPU (NDWI)
    #Fòrmula: NDWI = (Verd-NIR) /(Verd+NIR)
    denominador = (gpu_verda + gpu_nir)
    denominador[denominador == 0] = 0.0001 #Evitar la divisió per 0 per seguretat

    ndwi_gpu = (gpu_verda - gpu_nir) / denominador

    #NDWI > 0 és aigua
    mascara_aigua_gpu = ndwi_gpu > 0.15 #Si augmentem aquest valor, es fa més estricte i no es pensa que les ombres de núvols i muntanyes és aigua


    #4. EXTRACCIÓ DEL RESULTAT (Tornar només allò important a la Terra)
    #total_pixels_aigua = np.sum(mascara_aigua_gpu)
    total_pixels_aigua = np.count_nonzero(mascara_aigua_gpu)
    hectarees = float((total_pixels_aigua * area_pixel_m2) / 10000.0)

    #return mascara_aigua_gpu.get(), hectarees --> for CuPY
    return mascara_aigua_gpu, hectarees



#--------------------PROCESSAMENT DE LES IMATGES-----------------
def processar_directori(carpeta_imatges):
    #Llegeix totes es imatges d'una carpeta, les processa a la GPU i retorna una llista amb l'evolució de l'aigua al llarg del temps.

    resultats = []
    print("Iniciant processament amb la GPU de la carpeta: " + str(carpeta_imatges))

    #busquem tots els arxius .tif de la carpeta
    arxius = [f for f in os.listdir(carpeta_imatges) if f.endswith('.tif')]

    for arxiu in arxius:
        ruta_completa = os.path.join(carpeta_imatges, arxiu)

        #Per cada imatge cridem a la funció principal
        mascara, hectarees = processar_imatge_aigua(ruta_completa)

        nom_png = arxiu.replace('.tif', '_mask.png')
        ruta_png = os.path.join(carpeta_imatges, nom_png)

        plt.imsave(ruta_png, mascara, cmap = 'Blues')

        #agafem la data en la que sha fet la foto de la imatge (els 8 primers caràcters del nom)
        data_crua = arxiu[:8]
        data_neta = str(data_crua[6:8])+"/"+str(data_crua[4:6])+"/"+str(data_crua[:4]) # ho passem a format '08/04/2026

        #Guardem el nomde l'arxiu i de les hectàrees calcualdes
        #és un diccionari que l'afegim a una llista
        dic = {'arxiu': arxiu,'imatge_png': nom_png, 'data': data_neta, 'hectarees': hectarees}
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

import ee
import geemap
import os
from datetime import datetime
from functools import reduce

# Nombre màxim d'imatges a descarregar en mode normal (interval seleccionat). No es tria per núvols,
# només per no descarregar-ne un nombre excessiu; és la IA la que després descarta les nuvoloses.
LIMIT_IMATGES_MODE_NORMAL = 30

# Mode històric: nombre màxim d'imatges per finestra (Febrer o Agost) DE CADA ANY.
# Ho limitem per finestra i no de forma global perquè un .limit() global, un cop ordenat per data,
# es quedaria només amb els primers anys (p.ex. 2017-2019) i descartaria tots els anys més recents.
# Amb 3 per finestra tenim marge: si la 1a surt massa nuvolosa, la IA encara té 2 alternatives d'aquell mes.
LIMIT_IMATGES_PER_PERIODE_HISTORIC = 3

#
#-----------FUNCIÓ PER INICIALITZAR GOOGLE EARTH ENGINE ------------------------
def inicialitzar_gee():
    try:
       # Definim la ruta de l'arxiu JSON que acabes de posar al projecte
       # (a Docker es pot muntar la clau com a volum i indicar-ne la ruta amb GEE_CREDENTIALS)
       ruta_clau = os.environ.get('GEE_CREDENTIALS') or os.path.join(os.path.dirname(__file__), 'credentials2.json')

       # Correu electrònic de la Service Account que has copiat al Pas 1 (canvia-ho pel teu!)
       email_bot = 'visor-water-bsc@ee-guillemsadurnif.iam.gserviceaccount.com'
       ###

       # Inicialitzem amb les credencials del bot (Servei automatitzat 24/7)
       creds = ee.ServiceAccountCredentials(email_bot, ruta_clau)
       ee.Initialize(creds, project='ee-guillemsadurnif')

       print("Connexió amb Google Earth Engine establerta correctament amb Service Account")
    except Exception as e:
        print("No s'ha pogut establir la connexió amb Google Earth Engine.", e)
        raise


#-----------FUNCIÓ PER SIMULAR CÀMARA SATÈL·LIT------------------------
#Simula la càmera del satèl·lit. Descarreguem les dades en brut d'un àrea concreta sense processar.
def extreure_imatges_satelit(bbox, data_ini, data_fin,dir_sortida, mode_historic = False): # quan passem un parametre amb nom = valor, és un valor per defecte
    #bbox: llista amb les coord[lon_min, lat_min, lon_max, lat_max]
    #data_ini: ex:'2025-01-01'
    #dir_sortida: Ruta on guardar els arxius de les imatges .tif

    geo_desitjada = ee.Geometry.Rectangle(bbox)

    if mode_historic:#Si s'ha seleccionat la casella de l'històric
        # Aquí tampoc filtrem per CLOUDY_PIXEL_PERCENTAGE: la IA de obpmark_ml és qui detecta i descarta
        # les imatges massa nuvoloses un cop descarregades (a gpu_processing.processar_imatge_aigua).
        #
        # Construïm la col·lecció ANY PER ANY i FINESTRA PER FINESTRA (Febrer / Agost), limitant cada
        # finestra per separat (LIMIT_IMATGES_PER_PERIODE_HISTORIC). Així cada any queda representat
        # per igual: un .limit() global un cop ordenat per data només ens donaria els primers anys.
        any_inici = datetime.strptime(data_ini, '%Y-%m-%d').year
        any_final = datetime.strptime(data_fin, '%Y-%m-%d').year

        subcolleccions = []
        for any_ in range(any_inici, any_final + 1):
            for mes in (2, 8): #Febrer i Agost
                inici_finestra = f"{any_}-{mes:02d}-01"
                fi_finestra = f"{any_}-{mes+1:02d}-01" #mes+1 és vàlid: 3 (març) o 9 (setembre)
                sub = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(geo_desitjada)
                       .filterDate(inici_finestra, fi_finestra)
                       .sort('system:time_start')
                       .limit(LIMIT_IMATGES_PER_PERIODE_HISTORIC))
                subcolleccions.append(sub)

        #Ajuntem totes les finestres en una sola col·lecció i la tornem a ordenar per data
        colleccio = (reduce(lambda a, b: a.merge(b), subcolleccions)
                     .sort('system:time_start')
                     .select(['B2', 'B3', 'B4','B8'])
                     #B2(blau), B3(verd), B4(vermell): RGB per poder veure el mapa vista real
                     #B8(NIR - Near Infrared): Per detectar l'aigua
                     )
    else:#si no s'ha seleccionat la casella de l'historic
        # --- FILTRE NORMAL (Interval seleccionat) ---
        # NO filtrem ni ordenem per CLOUDY_PIXEL_PERCENTAGE (és el % de núvols que calcula Google, no nosaltres):
        # volem les imatges crues tal com les capta el satèl·lit. És la IA de obpmark_ml (a gpu_processing.py)
        # qui ha de decidir, un cop descarregada cada imatge, si té massa núvols per ser útil.
        colleccio = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(geo_desitjada)
                     .filterDate(data_ini, data_fin)
                     .sort('system:time_start') # ordenem per data de captura (no per núvols)
                     .limit(LIMIT_IMATGES_MODE_NORMAL)
                     .select(['B2', 'B3', 'B4','B8'])
                     )
    
    num_imatges = colleccio.size().getInfo()

    if num_imatges == 0:
        print("No s'ha trobat cap imatge vàlida per aquestes dates i coordenades")
        return False

    print("S'han trobat "+ str(num_imatges)+' vàlides.')

    #Creem la carpeta si no existeix
    if not os.path.exists(dir_sortida):
        os.makedirs(dir_sortida)

    #Descarreguem els arxius localment
    try:
        geemap.ee_export_image_collection(
            colleccio,
            out_dir = dir_sortida,
            scale = 10,
            region = geo_desitjada
        )
        # geemap no llança excepció si una imatge concreta falla: comprovem que s'hagi baixat alguna cosa
        n_descarregades = len([f for f in os.listdir(dir_sortida) if f.endswith('.tif')])
        if n_descarregades == 0:
            print("La descàrrega no ha generat cap arxiu .tif (revisa la mida de la zona o la connexió).")
            return False

        print(f"Descàrrega completada ({n_descarregades}/{num_imatges} imatges). Els arxius estan desats a " + str(dir_sortida))
        return True

    except Exception as e:
        print("Error durant la descàrrega:",e)
        return False





if __name__ == "__main__":
    inicialitzar_gee()

    bbox_prova = [2.370, 41.955, 2.435, 42.000]
    ruta_descarrega = os.path.join(os.getcwd(),'dades_satelit_crues')

    extreure_imatges_satelit(bbox_prova, '2026-01-01', '2026-06-20', ruta_descarrega)
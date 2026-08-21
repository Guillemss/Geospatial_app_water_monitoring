
import ee
import geemap
import os

#
#-----------FUNCIÓ PER INICIALITZAR GOOGLE EARTH ENGINE ------------------------
def inicialitzar_gee():
    try:
       ee.Initialize(project='ee-guillemsadurnif') # <--- AQUESTA ÉS LA LÍNIA CLAU QUE ET FALTAVA
       print("Connexió amb Google Earth Engine establerta correctament")
    except Exception as e: #guardem l'error que surti a la variable e
        print("No s'ha pogut establir la connexió amb Google Earth Engine.", e)
        raise #raise mostra l'error i atura el programa



#-----------FUNCIÓ PER SIMULAR CÀMARA SATÈL·LIT------------------------
#Simula la càmera del satèl·lit. Descarreguem les dades en brut d'un àrea concreta sense processar.
def extreure_imatges_satelit(bbox, data_ini, data_fin,dir_sortida, mode_historic = False): # quan passem un parametre amb nom = valor, és un valor per defecte
    #bbox: llista amb les coord[lon_min, lat_min, lon_max, lat_max]
    #data_ini: ex:'2025-01-01'
    #dir_sortida: Ruta on guardar els arxius de les imatges .tif

    geo_desitjada = ee.Geometry.Rectangle(bbox)

    if mode_historic:#Si s'ha seleccionat la casella de l'històric
        filtre_mesos = ee.Filter.Or(
            ee.Filter.calendarRange(2,2,'month'), #Febrer
            ee.Filter.calendarRange(8,8,'month') #Agost
        )
        colleccio = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(geo_desitjada)
                    .filterDate(data_ini, data_fin)
                    #Filtrem les imatges amb masses núvols --> Això es podria fer amb la IA del Jannis a la GPU
                    .filter(filtre_mesos)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                    #--------------!!!!!!!VIGILAR PERQUE QUAN FEM EL SORT, LES IMATGES DEIXEN D'ESTAR ORDENADES PER LA DATA EN LA QUE SHA FET LA FOTO!!!-------------
                    #.sort('CLOUDY_PIXEL_PERCENTAGE')#ordenem les imatges pel percentatge de núvols
                    #.limit(30) #Seleccionem les 20 imatges que tinguin més bon percentatge de visibilitat, sense núvols!
                    #Seleccionem les bandes
                    .select(['B2', 'B3', 'B4','B8' ])
                    #B2(blau), B3(verd), B4(vermell): RGB per poder veure el mapa vista real
                    #B8(NIR - Near Infrared): Per detectar l'aigua
                    )
    else:
        # --- FILTRE NORMAL (Interval seleccionat) ---
        colleccio = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(geo_desitjada)
                     .filterDate(data_ini, data_fin)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                     .sort('CLOUDY_PIXEL_PERCENTAGE')
                     .limit(20)
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
        print("Descàrrega completada. Els arxius estan desat a " + str(dir_sortida))
        return True

    except Exception as e:
        print("Error durant la descàrrega:",e)
        return False





if __name__ == "__main__":
    inicialitzar_gee()

    bbox_prova = [2.370, 41.955, 2.435, 42.000]
    ruta_descarrega = os.path.join(os.getcwd(),'dades_satelit_crues')

    extreure_imatges_satelit(bbox_prova, '2026-01-01', '2026-06-20', ruta_descarrega)
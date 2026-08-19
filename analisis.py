import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw #Per poder escriure la data de la foto a la imatge del vídeo
import os

def generar_grafic_evolucio(resultats):
    #Aquesta funció reb les dades calculades i genera la gràfica d'evolució
    
    #agafem els arxius i hectàrees de la llista de diccionaris
    data_arxius = [r['data'] for r in resultats]
    hectarees = [r['hectarees'] for r in resultats]

    #DIBUIXAR EL GRÀFIC amb Matplotlib
    fig, ax = plt.subplots(figsize = (10,4))

    ax.plot(data_arxius, hectarees, marker = 'o', color = 'teal', linewidth = 2, markersize=8)
    ax.set_ylabel("Hectàrees d'aigua (ha)")
    ax.set_xlabel("Imatges del Sentinel-2")
    ax.set_title("Evolució temporal de la zona seleccionada")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True)

    st.pyplot(fig)



def generar_timelapse(resultats, ruta_carpeta, ruta_sortida_gif):
    imatges_gif = []

    for i in resultats:
        if 'imatge_png' in i:
            ruta_png = os.path.join(ruta_carpeta, i['imatge_png'])

            if os.path.exists(ruta_png):
                img = Image.open(ruta_png).convert("RGBA") #ens assegurem que la imatge té 4 canals perque quan hi pintem a sobre, es vegin bé els colors
                #RGB + alpha (transparència)

                #dibuixar la data a sobre de la imatge
                draw = ImageDraw.Draw(img) #ens dona un pinzell virtual per pintar a sobre
                text_data = i['data']
                #Coordenades: (x_inici, y_inici), (x_final, y_final)#dibuixem un rectangle negre
                draw.rectangle([(10, 10), (90, 30)], fill="black")
                # Escrivim la data en blanc a sobre del rectangle
                draw.text((15, 15), text_data, fill="white")#escrivim la data en blanc
                
                # Guardem aquest "fotograma" a la llista
                imatges_gif.append(img)

    if imatges_gif:
        imatges_gif[0].save(
            ruta_sortida_gif,
            save_all = True,
            append_images = imatges_gif[1:],
            duration = 1000,
            loop = 0 #El 0 fa que el GIF es repeteixi en bucle infinit
        )
        return True

    return False
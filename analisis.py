import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw #Per poder escriure la data de la foto a la imatge del vídeo
import os
from datetime import datetime

def generar_grafic_evolucio(resultats):
    #Aquesta funció reb les dades calculades i genera la gràfica d'evolució
    
    if not resultats:
        st.info("No hi ha imatges vàlides per dibuixar el gràfic.")
        return

    #agafem les dates i hectàrees de la llista de diccionaris
    #(les dates es passen a datetime perquè l'eix X respecti el temps real entre imatges)
    data_arxius = [datetime.strptime(r['data'], "%d/%m/%Y") for r in resultats]
    hectarees = [r['hectarees'] for r in resultats]

    #DIBUIXAR EL GRÀFIC amb Matplotlib
    fig, ax = plt.subplots(figsize = (10,4))

    ax.plot(data_arxius, hectarees, marker = 'o', color = 'teal', linewidth = 2, markersize=8)
    ax.set_ylabel("Hectàrees d'aigua (ha)")
    ax.set_xlabel("Data de la imatge del Sentinel-2")
    ax.set_title("Evolució temporal de la zona seleccionada")
    fig.autofmt_xdate(rotation=45, ha='right')
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig) #alliberem la figura: aquesta funció s'executa a cada rerun de Streamlit

##provaaaaaa

def generar_grafic_evolucio_incendi(resultats):
    #Igual que generar_grafic_evolucio, però per la superfície CREMADA (acumulada des de la
    #imatge de referència), amb un color diferent per no confondre'l amb el gràfic de l'aigua.

    if not resultats:
        st.info("No hi ha imatges vàlides per dibuixar el gràfic.")
        return

    data_arxius = [datetime.strptime(r['data'], "%d/%m/%Y") for r in resultats]
    hectarees = [r['hectarees_cremades'] for r in resultats]

    fig, ax = plt.subplots(figsize = (10,4))

    ax.plot(data_arxius, hectarees, marker = 'o', color = 'firebrick', linewidth = 2, markersize=8)
    ax.fill_between(data_arxius, hectarees, color = 'firebrick', alpha = 0.15)
    ax.set_ylabel("Hectàrees cremades (acumulat des de la referència)")
    ax.set_xlabel("Data de la imatge del Sentinel-2")
    ax.set_title("Evolució del perímetre cremat")
    fig.autofmt_xdate(rotation=45, ha='right')
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)


def generar_graella_escaneig(resultats_graella, n_files, n_cols):
    #Dibuixa la graella de tessel·les escanejades: cadascuna amb la seva vista real i un color de
    #vora segons la classificació (verd=sense canvi, vermell=prioritat alta, gris=sense dades útils).
    #Aquesta és la simulació de "el satèl·lit escaneja una franja sense saber a priori on hi ha res
    #interessant", en lloc de dir-li directament les coordenades exactes de l'incendi.
    colors = {
        'prioritat_alta': '#dc3545',   # vermell
        'sense_canvi': '#28a745',      # verd
        'sense_dades': '#6c757d',      # gris
    }
    etiquetes = {
        'prioritat_alta': '🔥 PRIORITAT ALTA',
        'sense_canvi': '✅ Sense canvi',
        'sense_dades': '⚠️ Sense dades útils',
    }

    fig, axs = plt.subplots(n_files, n_cols, figsize=(4.2 * n_cols, 4.2 * n_files), squeeze=False)
    # Fila 0 = la de més al nord (lat_max); a la graella les files es numeren de sud a nord, així que
    # les invertim aquí només per mostrar-les amb el nord amunt, com un mapa.
    for r in resultats_graella:
        ax = axs[n_files - 1 - r['fila']][r['col']]
        if r['rgb_png'] and os.path.exists(r['rgb_png']):
            ax.imshow(Image.open(r['rgb_png']))
        else:
            ax.set_facecolor('#eeeeee')
        color = colors.get(r['classificacio'], '#6c757d')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
        titol = etiquetes.get(r['classificacio'], r['classificacio'])
        if r['hectarees_cremades'] is not None:
            titol += f"\n{r['hectarees_cremades']:.1f} ha"
        ax.set_title(titol, fontsize=10, color=color)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def generar_timelapse(resultats, ruta_carpeta, ruta_sortida_gif, clau_imatge = 'imatge_png'):
    #clau_imatge: quina imatge de cada resultat es fa servir per als fotogrames.
    #Per defecte 'imatge_png' (la màscara d'aigua/NDWI, blava sobre blanc, es llegeix bé).
    #En mode incendis convé passar 'rgb_png': la màscara de cremat és binària i quan l'incendi
    #només ocupa una part petita del requadre, el timelapse surt gairebé tot negre i no s'hi veu res.
    imatges_gif = []

    for i in resultats:
        if clau_imatge in i:
            ruta_png = os.path.join(ruta_carpeta, i[clau_imatge])

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
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


def generar_grafic_comparacio_temps(temps_bord_s, temps_baixada_s, temps_terra_s, amplada_banda_mbps):
    #Compara el temps real de processament a bord amb una estimació il·lustrativa de "baixar-ho tot a
    #terra i processar-ho allà". Nomia les dues barres: la primera és una mesura real (aquesta
    #execució); la segona és una estimació (mida real de les dades ÷ amplada de banda assumida, més el
    #mateix temps de processament que ja hem mesurat, assumint maquinari equivalent a terra).
    fig, ax = plt.subplots(figsize=(8, 3.2))
    etiquetes = ["A bord\n(GPU, mesurat)", "Baixar-ho tot a Terra\n(estimat)"]
    valors = [temps_bord_s, temps_baixada_s + temps_terra_s]
    colors = ["#28a745", "#dc3545"]
    barres = ax.barh(etiquetes, valors, color=colors)
    for barra, valor in zip(barres, valors):
        ax.text(valor, barra.get_y() + barra.get_height()/2, f"  {valor:.1f} s", va='center', fontsize=11)
    ax.set_xlabel("Segons")
    ax.set_title(f"Temps total: processar a bord vs. baixar-ho tot primer (enllaç il·lustratiu de {amplada_banda_mbps:g} Mbps)")
    ax.set_xlim(0, max(valors) * 1.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def generar_grafic_estalvi_acumulat(resultats):
    #Mostra com creix l'estalvi de dades AL LLARG de la missió, no només com un únic número final.
    #Dues línies a la MATEIXA escala (MB) perquè la diferència es vegi de veritat: la de "si es
    #baixés tot" creix imatge a imatge; la de "només les alertes prioritzades" es queda gairebé
    #plana, perquè 1 KB per alerta és insignificant comparat amb els MB d'una imatge sencera.
    if not resultats:
        st.info("No hi ha imatges vàlides per dibuixar el gràfic.")
        return

    dates = [datetime.strptime(r['data'], "%d/%m/%Y") for r in resultats]
    mb_acumulat, mb_alertes_acumulat = [], []
    total_mb = 0.0
    total_kb_alertes = 0.0
    for r in resultats:
        total_mb += r.get('mida_bytes', 0) / 1024 / 1024
        if r.get('alerta_creixement', False):
            total_kb_alertes += 1.0  # 1 KB il·lustratiu per alerta (coordenades + hectàrees), com al panell d'estalvi
        mb_acumulat.append(total_mb)
        mb_alertes_acumulat.append(total_kb_alertes / 1024)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, mb_acumulat, marker='o', color='#dc3545', linewidth=2, label="Si es baixés cada imatge sencera")
    ax.fill_between(dates, mb_acumulat, color='#dc3545', alpha=0.08)
    ax.plot(dates, mb_alertes_acumulat, marker='o', color='#28a745', linewidth=2, label="Només les alertes prioritzades")
    ax.set_ylabel("Dades acumulades (MB)")
    ax.set_xlabel("Data de la imatge del Sentinel-2")
    ax.set_title("Estalvi de dades acumulat al llarg de la missió")
    fig.autofmt_xdate(rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
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
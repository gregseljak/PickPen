    # PickPen
    # Gregory Seljak 2022
    ######################################################################
    # Ceci est mon projet pour developper mes abilites en machine learning
    # et traitement de signal. Le but des modèles est de récuperer la partition 
    # musicale d'un fichier .wav. Voici le resume de la module:

    1. pkpn_synthgen.py
        - creer des échantillons synthétiques pour entrainer et aussi évaluer
        le modèle.
        - les "partitions" sont stockées dans le fichier ydata.npz en format
        np.array([], dtype=(time_on, time_off, pitch, vol))
        -- valeurs de temps en ms
    
    2. pkpn_learn.py
        - procédure de l'entrainement (selection de l'architecture,
        hyperparametres, etc.)

    3. wav_preprocess.py
        - préparation des segments du signal pour rendre des échantillons de
        longeurs uniformes au modèles.
        - Actuellement aussi fait la distinction [fenetre visible / fenetre sensible]
            ce qui devrait etre un attribut du modele (à regler plus tard)
    
    4. pkpn_evaluate.py
        - simulation de la vraie exécution du produit
        -- Chaque ségment est comparé à la partition
        -- (réussite/échec) de la catégorisation


    --------------------------------- divers --------------------
    5. toml_config.py
        - interface pour faciliter le module argparse avec des fichiers de configuration .toml

    6. model_library.py
        - stockage des archictures pour qu'elles soient disponibles à pkpn_train.py

    7. csv2midi.sh
        - scripte pour transformer la partition de .csv en .wav:
        -- .csv -> .mid avec le module midicsv: https://www.fourmilab.ch/webtools/midicsv/#Download
            # Actuellement j'utilise le format windows .exe, plus tard je changerai à linux
        -- .mid -> .wav avec le module timidity: https://doc.ubuntu-fr.org/timidity
            # Il faut aussi un soundfont: https://www.producerfeed.com/2010/02/49-guitar-soundfonts.html
        -- .wav -> .wav avec le module ffmpeg:  https://www.ffmpeg.org/download.html
            # Ajustement final comme timidity ne respecte pas ni la durée ni le volume du signal

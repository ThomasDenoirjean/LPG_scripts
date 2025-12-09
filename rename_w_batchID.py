import os

def renommer_fichiers(dossier, id_fixe):
    """
    Renomme tous les fichiers du dossier en ajoutant un ID fixe devant leur nom,
    avec l'extension .png.

    Args:
        dossier (str): Chemin du dossier contenant les fichiers à renommer.
        id_fixe (str): ID fixe à ajouter devant chaque nom de fichier.
    """
    # Vérifie que le dossier existe
    if not os.path.exists(dossier):
        print(f"Le dossier '{dossier}' n'existe pas.")
        return

    # Liste tous les fichiers dans le dossier
    fichiers = os.listdir(dossier)

    for fichier in fichiers:
        if fichier.endswith('.png'):
            # Construit les chemins complets
            ancien_chemin = os.path.join(dossier, fichier)
            # Ignore les dossiers, ne traite que les fichiers
            if os.path.isfile(ancien_chemin):
                nouveau_nom = f"{id_fixe}_{fichier}"
                nouveau_chemin = os.path.join(dossier, nouveau_nom)

                # Renomme le fichier
                os.rename(ancien_chemin, nouveau_chemin)
                print(f"Renommé: {fichier} -> {nouveau_nom}")

# Exemple d'utilisation
dossier = "."  # Remplace par le chemin de ton dossier
id_fixe = "20250717_102242"                   # Remplace par l'ID fixe souhaité
renommer_fichiers(dossier, id_fixe)

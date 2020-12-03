import argparse
from utils import lire_json, predir
from model import ChargerModel

# passer des paramètres au lancement du script d'apprentissage
parser = argparse.ArgumentParser(description="Prédire l'image avec le modèle de classificateur")
parser.add_argument("img_path", help="définir le chemin vers l'image")
parser.add_argument("checkpoint", help="définir le chemin vers le point de sauvegarde de d'apprentisage")
parser.add_argument("--top_k", type=int, default=1, help="Selectionner le top K des predictions")
parser.add_argument("--category_names", help="choisir les noms des catégories")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="gpu utilisé")

# parser les arguments
args = parser.parse_args()

# si on a défini une catégorie dans les arguments
if args.category_names:
    #je récupère la catégorie dans les json des libellé de fleur
    cat_to_name = lire_json(args.category_names)
else: 
    #sinon je le passe à rien
    cat_to_name = None

# je recupère les backup des model run précedement 
model_cp = ChargerModel(args.checkpoint)

# essaye de predire l'image  passer en paramettre avec le facteur topk et le non de la catégorie s'il y en a
predir(image_path=args.img_path, model=model_cp, topk=args.top_k, device=args.gpu, cat_to_name=cat_to_name)
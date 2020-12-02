import argparse
from utils_ic import lire_json, predir
from model_ic import chargerModel

# passer des paramètres au lancement du script d'apprentissage
parser = argparse.ArgumentParser(description="Predict image with classifier model")
parser.add_argument("img_path", help="set path to image")
parser.add_argument("checkpoint", help="set path to checkpoint")
parser.add_argument("--top_k", type=int, default=1, help="Select top K predictions")
parser.add_argument("--category_names", help="choose category names")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")

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
model_cp = chargerModel(args.checkpoint)

# essaye de predire l'image  passer en paramettre avec le facteur topk et le non de la catégorie s'il y en a
predir(image_path=args.img_path, model=model_cp, topk=args.top_k, device=args.gpu, cat_to_name=cat_to_name)
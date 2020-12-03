import argparse
from utils import traitement_image, lire_json
from model import classificationReseauNeuronne, validation, creerReseauNeuronne, sauvegarderModel

#passer des paramètres au lancement du script d'apprentissage
parser = argparse.ArgumentParser(description="Modèle de classificateur d'images de l'entrainement")
parser.add_argument("data_dir", help="le répertoire de données")
parser.add_argument("--category_names", default="cat_to_name.json", help="choisir les noms des catégories")
parser.add_argument("--arch", default="densenet169", help="choisir l'architecture du modèle")
parser.add_argument("--learning_rate", type=int, default=0.001, help="définir le taux d'apprentissage")
parser.add_argument("--hidden_units", type=int, default=1024, help="définir des unités cachées")
parser.add_argument("--epochs", type=int, default=1, help="définir epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="gpu utilisé")
parser.add_argument("--save_dir", help="dossier où enregistrer le modèle")

#parser les arguments
args = parser.parse_args()

#convertir et lire les données en fonction des catégories de fleurs choisi
cat_to_name = lire_json(args.category_names)

#dossiers contenant les images
trainloader, testloader, validloader, train_data = traitement_image(args.data_dir)

#initialise le reseau de neuronnes
model = creerReseauNeuronne(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)

#si un dossier a été choisi pour récupérer les models sauvegarder, je le sauvegarde
if args.save_dir:
    sauvegarderModel(model, args.save_dir)
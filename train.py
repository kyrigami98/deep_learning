import argparse
from utils_ic import chargerImages, read_jason
from model_ic import classificationReseauNeuronne, validation, creerReseauNeuronne, sauvegarderModel

#passer des paramètres au lancement du script d'apprentissage
parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")

#parser les arguments
args = parser.parse_args()

#convertir et lire les données en fonction des catégories de fleurs choisi
cat_to_name = read_jason(args.category_names)

#dossiers contenant les images
trainloader, testloader, validloader, train_data = chargerImages(args.data_dir)

#initialise le reseau de neuronnes
model = creerReseauNeuronne(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)

#si un dossier a été choisi pour récupérer les models sauvegarder, je le sauvegarde
if args.save_dir:
    sauvegarderModel(model, args.save_dir)
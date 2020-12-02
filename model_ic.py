import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

# classifieur
class classificationReseauNeuronne(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        '''         

        Un réseau de neurones se compose d'unités (neurones), disposées en couches, 
        qui convertissent un vecteur d'entrée en une sortie. 
        Chaque unité prend une entrée, lui applique une fonction (souvent non linéaire), 
        puis transmet la sortie à la couche suivante.

        Un réseau de neurones feedforward est un algorithme de classification d'inspiration biologique . 
        Il se compose d'un (peut-être grand) nombre d' unités de traitement simples de type neurone , organisées en couches. 
        Chaque unité d'une couche est connectée à toutes les unités de la couche précédente. ...
        C'est pourquoi ils sont appelés réseaux de neurones feedforward .

        Construit un réseau d'anticipation avec des couches cachées arbitraires.
        
        Arguments
        ---------
            input_size: entier, taille de l'entrée
            output_size: entier, taille de la couche de sortie
            hidden_layers: liste d'entiers, les tailles des couches cachées
            drop_p: flottant entre 0 et 1, probabilité d'abandon

        '''
        super().__init__()
        # Ajouter la première couche, entrée dans la couche caché
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Ajouter un nombre variable de couches plus cachées
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def next(self, x):
        ''' se déplacer en avant à travers le réseau de neurone, renvoie les neuronnes de sortie '''
        
        # se déplacer à travers chaque couche dans `hidden_layers`, avec l'activation et la suppression de ReLU(Rectified linear units)
        """ 
        La fonction d'activation linéaire redressée ou ReLU en abrégé est 
        une fonction linéaire par morceaux qui sortira directement l'entrée si elle est positive, 
        sinon elle produira zéro. 

        """
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

# Define validation function 
def validation(model, testloader, criterion, device):

"""
Les ensembles de données de validation peuvent être utilisés pour la régularisation par arrêt anticipé 
(arrêt de la formation lorsque l'erreur sur l'ensemble de données de validation augmente)

"""
    # La valeur de perte implique dans quelle mesure un certain modèle se comporte bien ou mal après chaque itération d'optimisation. 
    # Idéalement, on s'attendrait à une réduction de la perte après chaque ou plusieurs itérations.
    test_loss = 0

    # précision
    # La précision d'un modèle est généralement déterminée après que les paramètres du modèle ont été appris et fixés et qu'aucun apprentissage n'a lieu.
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.next(images)
        test_loss += criterion(output, labels).item()
        
        # exp renvoie un nouveau tenseur avec l'exponentiel des éléments du tenseur d'entrée
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Define creer un reseau de neuronne
def creerReseauNeuronne(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data):
    # Importer le modèle de reseau de neuronne pré-entraîné
    model = getattr(models, model_name)(pretrained=True)
    
    # geler une partie de votre modèle et entraîner le reste
    for param in model.parameters():
        param.requires_grad = False
        
    # faire la classification 
    n_in = next(model.classifier.modules()).in_features
    n_out = len(labelsdict) 
    model.classifier = classificationReseauNeuronne(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)
    
    # critère 
    criterion = nn.NLLLoss()
    # #un modèle d'optimisation se compose d'une fonction objectif (également appelée critère d'optimisation ou fonction d'objectif) et de contraintes. 
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

    model.to(device)

    # temps de lancement
    start = time.time()

    # Nombre d'epochs à exécuter.
    epochs = n_epoch 
    steps = 0 
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.next(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Mode d'évaluation pour les prédictions
                model.eval()

                # Désactivez les gradients pour validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Perte d'entraînement: {:.3f} - ".format(running_loss/print_every),
                      "Perte lors de la validation: {:.3f} - ".format(test_loss/len(validloader)),
                      "Précision de la validation: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # entrainer le model
                model.train()
    
    # ajouter les informations de model 
    model.classifier.n_in = n_in
    model.classifier.n_hidden = n_hidden
    model.classifier.n_out = n_out
    model.classifier.labelsdict = labelsdict
    model.classifier.lr = lr
    model.classifier.optimizer_state_dict = optimizer.state_dict
    model.classifier.model_name = model_name
    model.classifier.class_to_idx = train_data.class_to_idx
    
    # affichage du résult
    print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return model

# sauvegarder le Model pour la prédiction
def sauvegarderModel(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)
    
# Define function to load model
def ChargerModel(path):
    cp = torch.load(path)
    
    # Importer le modèle de reseau de neuronne pré-entraîné
    model = getattr(models, cp['model_name'])(pretrained=True)
    
    # geler une partie de votre modèle et entraîner le reste
    for param in model.parameters():
        param.requires_grad = False
    
    # faire la classification
    model.classifier = classificationReseauNeuronne(input_size=cp['c_input'], output_size=cp['c_out'], \
                                     hidden_layers=cp['c_hidden'])
    
    # ajouter les informations de model
    model.classifier.n_in = cp['c_input']
    model.classifier.n_hidden = cp['c_hidden']
    model.classifier.n_out = cp['c_out']
    model.classifier.labelsdict = cp['labelsdict']
    model.classifier.lr = cp['c_lr']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])
    
    return model

#tester le model
def test_model(model, testloader, device='cuda'):  
    model.to(device)
    model.eval()
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.next(images)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))
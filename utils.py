import json
import torch
from torchvision import datasets, transforms
from PIL import Image

# lire les noms de fleur dans les fichiers
def lire_json(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

# lire les données dans les fichiers d'images
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Définir les transformations pour les ensembles d'images
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Charger les ensembles de données avec ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    # À l'aide des jeux de données d'image et des transformations, on définie les chargeurs de données
    # avec des batch_size qui represente le nombre d'element par training
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, testloader, validloader, train_data

# On utilise pillow pour rogner et redimmentionner pour le model PyTorch et retourne un tableau numpy
def traitement_image(image):

    # On utilise pillow pour rogner et redimmentionner
    im = Image.open(image)
    
    preprocess = transforms.Compose([transforms.Resize(255), 
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    im_tensor = preprocess(im)
    im_tensor.unsqueeze_(0)
    
    return im_tensor

# Prédire le model le plus probable 
def predir(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    img = traitement_image(image_path)
    img = img.to(device)
    
    output = model.forward(img)
    ps = torch.exp(output)    
    probabilities, idxs = ps.topk(topk)

    idx_to_class = dict((v,k) for k, v in model.classifier.class_to_idx.items())
    classes = [v for k, v in idx_to_class.items() if k in idxs.to('cpu').numpy()]
    
    if cat_to_name:
        classes = [cat_to_name[str(i + 1)] for c, i in \
                     model.classifier.class_to_idx.items() if c in classes]

    # on affiche une graphe et une image de la fleur trouver
    """ imshow(process_image(im_path).numpy()[0])
    plt.axis('off')

    fig, ax = plt.subplots()
    ax.barh(np.arange(5), probabilities)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels([model_cp.classifier.labelsdict[str(i + 1)] for c, i in \
                        model_cp.classifier.class_to_idx.items() if c in classes], size='small')
    plt.tight_layout()  
    """
        
    print('Probabilities:', probabilities.data.cpu().numpy()[0].tolist())
    print('Classes:', classes)
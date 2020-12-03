## Projet de Deep Learning - Reconnaissance de fleurs

### Résumé
Ce projet a pour but d'identifier une image et la classifier selon son type de fleurs.

### Pré-requis
- Installation de conda

- Installation de pyTorch

  `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

- OR install PyTorch using pip:

  `pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-win_amd64.whl`
  `pip3 install torchvision`
  
- Création de l'environement

  `conda create -n env_pytorch python=3.6`

- Activate the environment using:

  `conda activate env_pytorch`

### Execution

- Lancer l'entrainement 

`python train.py flowers`



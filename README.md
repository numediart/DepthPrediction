# Depth Prediction

Ce dépôt présente le modèle monodepth2 destiné à la prédiction de cartes de profondeur à partir d'images RGB. 
L'implémentation officielle du modèle est disponible à [cette adresse](https://github.com/nianticlabs/monodepth2). L'algorithme est présenté dans ["Digging Into Self-Supervised Monocular Depth Estimation"](https://arxiv.org/abs/1806.01260).

Monodepth2 apprend à prédire la profondeur de manière non supervisée à partir de séquences d'images monoculaires. Pour ce faire, il minimise l'erreur photométrique entre une image originale à un instant t et des images synthétisées à partir:
* des images aux instants t-1 et t+1,
* de la profondeur de la scène, et
* du mouvement de la caméra entre les instants t-1, t et t+1.

Il est aussi possible d'entraîner le modèle à partir de séquences stereo. 

### Requirements

* Python 3.6
* Ubuntu 18.04
* CUDA 9.1
* PyTorch 0.4.1

Les outils nécessaires peuvent être installés dans un environnement Anaconda à l'aide des commandes suivantes: 
```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
### Training

Le modèle peut être entraîné sur deux datasets: [Kitti](http://www.cvlibs.net/datasets/kitti/) et [Cityscapes](https://www.cityscapes-dataset.com/). Pour ce dernier, les images contenues dans les archives "leftImg8bit_sequence_trainvaltest.zip" et "rightImg8bit_sequence_trainvaltest.zip" sont nécessaires. Dans les deux cas, le modèle s'attend à recevoir des images au format `jpg`. Les images des deux datasets peuvent être converties à l'aide de la commande suivante: 
```shell
find dataset_root/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
Les commandes suivantes permettent d'entraîner le modèle sur Kitti et Cityscapes respectivement: 
```shell
python train.py --dataset kitti --data_path path/to/kitti/ \
--model_name your_model --log_dir path/to/log/file

python train.py --dataset cityscapes --data_path path/to/cityscapes/ \
--model_name your_model --split cityscapes --log_dir path/to/log/file
```
D'autres options peuvent être ajoutées, pour obtenir une liste complète, il suffit d'exécuter:
```shell 
python train.py -h
```
### Evaluation

L'evaluation du modèle peut se faire sur Kitti selon la procédure indiquée sur le [dépôt officiel](https://github.com/nianticlabs/monodepth2#-kitti-evaluation) de monodepth2. 

### Prediction

Il est possible d'obtenir une prédiction sur ses propres images en exécutant: 
```shell
python test_simple.py --image_path path/to/image.jpg --model_name model
```
La liste des noms de tous les modèles compatibles avec ce script est disponible sur le [dépôt officiel](https://github.com/nianticlabs/monodepth2/#%EF%B8%8F-prediction-for-a-single-image).

### License 

Copyright © Niantic, Inc. 2019. Patent Pending. All rights reserved. Please see the license file for terms.

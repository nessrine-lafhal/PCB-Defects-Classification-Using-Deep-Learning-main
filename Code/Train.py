from __future__ import print_function 
import keras
import cv2
import numpy as np
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization, Flatten
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling2D
from keras.regularizers import l2
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import math
#############################################################################################


# Fonction pour ajuster le taux d'apprentissage au fil des époques
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# Fonction pour créer le modèle DenseNet
def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
    
    # Vérifie que le nombre de classes a été défini, car il est nécessaire pour la couche finale softmax
    if nb_classes is None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')
    
    # Vérifie que le paramètre de compression est dans l'intervalle (0.0, 1.0]
    if compression <= 0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    
    # Si dense_layers est une liste, vérifie que sa longueur correspond au nombre de blocs denses
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    # Si dense_layers vaut -1, calcule le nombre de couches dans chaque bloc dense
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1)) / dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1)) // dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    # Si dense_layers est un entier, l'utilise pour chaque bloc dense
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
        
    # Crée l'entrée du modèle avec la forme spécifiée
    img_input = Input(shape=input_shape)
    # Le nombre de canaux initial est égal au taux de croissance multiplié par 2
    nb_channels = growth_rate * 2

    
   
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    # Ajoute un padding de zéro de 3 pixels de chaque côté de l'image d'entrée

    x = Conv2D(nb_channels, (7, 7), strides=2, use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    # Applique une couche de convolution avec un filtre de 7x7, un pas de 2, sans biais et avec une régularisation L2


    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    # Applique une normalisation par lots avec une régularisation L2 sur les paramètres gamma et beta

    x = Activation('relu')(x)
    # Applique une activation ReLU

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    # Ajoute un padding de zéro de 1 pixel de chaque côté

    
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # Applique une couche de max pooling avec une fenêtre de 3x3 et un pas de 2

    # Construction des blocs denses
    for block in range(dense_blocks):
        # Ajoute un bloc dense
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)
        
        if block < dense_blocks - 1:  # Si ce n'est pas le dernier bloc dense
            # Ajoute une couche de transition
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)
            # Met à jour le nombre de canaux en appliquant le facteur de compression

    # Pooling et couche finale
    x = AveragePooling2D(pool_size=7)(x)
    # Applique une couche de pooling moyen avec une fenêtre de 7x7
    x = Flatten(data_format='channels_last')(x)
    # Aplatie le tenseur en un vecteur
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    # Applique une couche dense (entièrement connectée) avec une activation softmax pour la classification finale
    # Utilise la régularisation L2 sur les poids et les biais


    # Nom du modèle pour indiquer la configuration
    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'  # Si le taux de croissance est élevé, le modèle est nommé 'widedense'
    else:
        model_name = 'dense'  # Sinon, il est nommé 'dense'

    if bottleneck:
        model_name = model_name + 'b'  # Si le modèle utilise des couches bottleneck, 'b' est ajouté au nom

    if compression < 1.0:
        model_name = model_name + 'c'  # Si une compression est appliquée, 'c' est ajouté au nom

    return Model(img_input, x, name=model_name), model_name  # Retourne le modèle et son nom

# Fonction pour créer un bloc dense
def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

    x_list = [x]  # Liste des entrées pour concaténation
    for i in range(nb_layers):  # Pour chaque couche dans le bloc dense
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)  # Crée un bloc de convolution
        x_list.append(cb)  # Ajoute le résultat à la liste
        x = Concatenate(axis=-1)(x_list)  # Concatène les résultats le long de l'axe des canaux
        nb_channels += growth_rate  # Augmente le nombre de canaux
    return x, nb_channels  # Retourne le bloc dense et le nombre de canaux

# Fonction pour créer un bloc de convolution
def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

    growth_rate = nb_channels / 2  # Taux de croissance défini comme la moitié du nombre de canaux

    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4  # Largeur du bottleneck
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)  # Normalisation par lots
        x = Activation('relu')(x)  # Activation ReLU
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)  # Convolution 1x1
        if dropout_rate:
            x = Dropout(dropout_rate)(x)  # Dropout

    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)  # Normalisation par lots
    x = Activation('relu')(x)  # Activation ReLU
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)  # Convolution 3x3
    if dropout_rate:
        x = Dropout(dropout_rate)(x)  # Dropout

    return x  # Retourne le bloc de convolution

# Fonction pour créer une couche de transition
def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)  # Normalisation par lots
    x = Activation('relu')(x)  # Activation ReLU
    x = Conv2D(int(nb_channels * compression), (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)  # Convolution 1x1

    if dropout_rate:
        x = Dropout(dropout_rate)(x)  # Dropout

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)  # Pooling moyen
    return x  # Retourne la couche de transition

if __name__ == '__main__':

    # Création et compilation du modèle
    model = DenseNet(input_shape=(64, 64, 1), dense_blocks=2, dense_layers=6, growth_rate=32, nb_classes=6, bottleneck=True, depth=27, weight_decay=1e-5)
    print(model[0].summary())  # Affiche le résumé du modèle
    opt = SGD(lr=0.0, momentum=0.9)  # Configuration de l'optimiseur SGD
    model[0].compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  # Compilation du modèle

    # Préparation des données d'entraînement
    train_datagen = ImageDataGenerator(data_format="channels_last")  # Générateur de données d'entraînement
    train_generator = train_datagen.flow_from_directory('TrainPath', target_size=(64, 64), color_mode='grayscale', batch_size=8)  
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size  # Calcul du nombre de pas par époque

    # Configuration des callbacks
    lrate = LearningRateScheduler(step_decay, verbose=1)  # Scheduler pour ajuster le taux d'apprentissage
    callbacks_list = [lrate]  # Liste des callbacks
    model[0].fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=25, callbacks=callbacks_list, verbose=1)  # Entraînement du modèle
    model[0].save("SavePath.keras")  # Sauvegarde du modèle

import os
from PIL import Image
import cv2

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf
from keras.losses import CategoricalCrossentropy


import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Reshape, Conv2DTranspose, MaxPool2D, Dense, Flatten, InputLayer, Lambda, Resizing
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

import albumentations as A

global_path = "data_sample_no_smarmellata/"

camera_path = global_path + "camera/"
label_path = global_path + "label/"

batch_size = 10
n_classi = 55



colori_classi_label_hex = {
    "#ff0000": "Car 1",
    "#c80000": "Car 2",
    "#960000": "Car 3",
    "#800000": "Car 4",
    "#b65906": "Bicycle 1",
    "#963204": "Bicycle 2",
    "#5a1e01": "Bicycle 3",
    "#5a1e1e": "Bicycle 4",
    "#cc99ff": "Pedestrian 1",
    "#bd499b": "Pedestrian 2",
    "#ef59bf": "Pedestrian 3",
    "#ff8000": "Truck 1",
    "#c88000": "Truck 2",
    "#968000": "Truck 3",
    "#00ff00": "Small vehicles 1",
    "#00c800": "Small vehicles 2",
    "#009600": "Small vehicles 3",
    "#0080ff": "Traffic signal 1",
    "#1e1c9e": "Traffic signal 2",
    "#3c1c64": "Traffic signal 3",
    "#00ffff": "Traffic sign 1",
    "#1edcdc": "Traffic sign 2",
    "#3c9dc7": "Traffic sign 3",
    "#ffff00": "Utility vehicle 1",
    "#ffffc8": "Utility vehicle 2",
    "#e96400": "Sidebars",
    "#6e6e00": "Speed bumper",
    "#808000": "Curbstone",
    "#ffc125": "Solid line",
    "#400040": "Irrelevant signs",
    "#b97a57": "Road blocks",
    "#000064": "Tractor",
    "#8b636c": "Non-drivable street",
    "#d23273": "Zebra crossing",
    "#ff0080": "Obstacles / trash",
    "#fff68f": "Poles",
    "#960096": "RD restricted area",
    "#ccff99": "Animals",
    "#eea2ad": "Grid structure",
    "#212cb1": "Signal corpus",
    "#b432b4": "Drivable cobblestone",
    "#ff46b9": "Electronic traffic",
    "#eee9bf": "Slow drive area",
    "#93fdc2": "Nature object",
    "#9696c8": "Parking area",
    "#b496c8": "Sidewalk",
    "#48d1cc": "Ego car",
    "#c87dd2": "Painted driv. instr.",
    "#9f79ee": "Traffic guide obj.",
    "#8000ff": "Dashed line",
    "#ff00ff": "RD normal street",
    "#87ceff": "Sky",
    "#f1e6ff": "Buildings",
    "#60458f": "Blurred area",
    "#352e52": "Rain dirt"
}

onehot_colori = {}
for idx, label in enumerate(colori_classi_label_hex.values(), start=0):
    onehot_colori[label] = idx

# faccio un sort in base ai values
onehot_colori = dict(sorted(onehot_colori.items(), key=lambda item: item[1]))

onehot_colori

strategy = tf.distribute.MirroredStrategy()

if __name__ == "__main__":
    print("Ottengo la lista di nomi dei file ...")
    if os.path.exists(camera_path):
        camera_imgs_list_names = os.listdir(camera_path)   #contengono i nomi delle singole immagini
        print("Lista nomi immagini popolata")
    else: raise Exception("Errore: cartella immagini non trovata")
    if os.path.exists(label_path):
        label_imgs_list_names = os.listdir(label_path)
        print("Lista nomi labels popolata")
    else: raise Exception("Errore: cartella label non trovata")
    #verifichiamo che il numero di immagini sia lo stesso
    if len(camera_imgs_list_names) == len(label_imgs_list_names):
        print("Il numero di immagini è lo stesso (",len(camera_imgs_list_names),")")
    else:
        raise Exception("Il numero di immagini è diverso")

    #verifichiamo che i nomi dei file siano uguali in entrambe le cartelle
    #immagine camera: 20180807145028_camera_frontcenter_000000091.png
    #immagine label: 20180807145028_label_frontcenter_000000091.png

    camera_imgs_list_names.sort()
    label_imgs_list_names.sort()
    for i in range(len(camera_imgs_list_names)):
        if camera_imgs_list_names[i].replace("camera", "") != label_imgs_list_names[i].replace("label", ""):
            raise Exception("I nomi dei file sono diversi")

    print("I nomi dei file sono uguali")

    print("\nTutto ok")


    def generatore_batch_immagini(camera_path, label_path, camera_imgs_list_names, label_imgs_list_names, batch_size):
        '''Generatore che restituisce un batch di immagini e le rispettive label

        Parametri
        ----------
        camera_path : string
            path della cartella contenente le immagini della camera
        label_path : string
            path della cartella contenente le immagini della label
        camera_imgs_list_names : list
            lista contenente i nomi delle immagini della camera
        label_imgs_list_names : list
            lista contenente i nomi delle immagini della label
        batch_size : int
            dimensione del batch di immagini che vogliamo ottenere

        Yields
        ------
        camera_batch : numpy array
            batch di immagini della camera
        label_batch : numpy array
            batch di immagini della label
        '''

        dimensione_output = (256,256)

        #shuffle delle liste

        for i in range(0, len(camera_imgs_list_names), batch_size):
            camera_batch = []
            label_batch = []
            for j in range(i, i + batch_size):
                with Image.open(camera_path + camera_imgs_list_names[j]) as camera:
                    camera = np.array(camera)
                    # Data augmentation, agendo solo sul colore, contrasto, luminosità saturazione

                    camera = tf.image.random_brightness(camera, 0.3)
                    camera = tf.image.random_contrast(camera, 0.8, 1.2)
                    camera = tf.image.random_saturation(camera, 0.8, 1.4)

                    #normalizzo i valori dei pixel
                    camera = camera / 255
                    # faccio si che la camera abbia shape (200, 318, 3)
                    camera = camera[:, :, :3]

                    # faccio il resize della camera
                    camera = tf.image.resize(camera, dimensione_output, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                with Image.open(label_path + label_imgs_list_names[j]) as label:
                    label = np.array(label)
                    #normalizzo i valori dei pixel
                    #label = label / 255
                    # faccio si che la label abbia shape (200, 355, 3)
                    label = label[:, :, :3]

                    #_______________________________________________________
                    # creo la label onehot, con shape (200, 355, 1) e valori da 0 a n_classi
                    # 0 = pixel macellati
                    # 1-n_classi = pixel corrispondenti alle classi
                    #_______________________________________________________

                    # Creo la label onehot

                    label_onehot = np.zeros((label.shape[0], label.shape[1], n_classi), dtype=np.uint8)
                    pixel_sbagliati = 0
                    # Converte i pixel da colore a encodig per classe
                    for row in range(label.shape[0]):
                        for col in range(label.shape[1]):
                            class_idx_trovato = None
                            pixel_color = tuple(label[row, col])
                            pixel_color = '#{:02X}{:02X}{:02X}'.format(pixel_color[0], pixel_color[1], pixel_color[2]).lower()
                            if str(pixel_color) in colori_classi_label_hex:
                                label_name = colori_classi_label_hex[pixel_color]
                                class_idx_trovato = onehot_colori[label_name]
                            for i in range(0, n_classi-1):
                                if i == class_idx_trovato:
                                    label_onehot[row, col, i] = 1
                                else:
                                    label_onehot[row, col, i] = 0

                    label = label_onehot

                    # faccio il resize della label
                    label = tf.image.resize(label, dimensione_output, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                camera_batch.append(camera)
                label_batch.append(label)
            yield np.array(camera_batch), np.array(label_batch)


    # train: 80%
    # validation: 10%
    # test: 10%

    n_immagini = len(camera_imgs_list_names)
    n_train = int(n_immagini * 0.8)
    n_val = int(n_immagini * 0.1)
    n_test = int(n_immagini * 0.1)

    # creo le liste dei nomi delle immagini
    train_camera_imgs_list_names = camera_imgs_list_names[:n_train]
    train_label_imgs_list_names = label_imgs_list_names[:n_train]

    val_camera_imgs_list_names = camera_imgs_list_names[n_train:n_train+n_val]
    val_label_imgs_list_names = label_imgs_list_names[n_train:n_train+n_val]

    test_camera_imgs_list_names = camera_imgs_list_names[n_train+n_val:]
    test_label_imgs_list_names = label_imgs_list_names[n_train+n_val:]

    # creo i dataset
    trainset = tf.data.Dataset.from_generator(
        generatore_batch_immagini,
        args=(camera_path, label_path, train_camera_imgs_list_names, train_label_imgs_list_names, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 256, 256, 3)),  # Immagine della camera
            tf.TensorSpec(shape=(batch_size, 256, 256, n_classi)),  # Immagine label
        )
    )

    valset = tf.data.Dataset.from_generator(
        generatore_batch_immagini,
        args=(camera_path, label_path, val_camera_imgs_list_names, val_label_imgs_list_names, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 256, 256, 3)),  # Immagine della camera
            tf.TensorSpec(shape=(batch_size, 256, 256, n_classi)),  # Immagine label
        )
    )

    testset = tf.data.Dataset.from_generator(
        generatore_batch_immagini,
        args=(camera_path, label_path, test_camera_imgs_list_names, test_label_imgs_list_names, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 256, 256, 3)),  # Immagine della camera
            tf.TensorSpec(shape=(batch_size, 256, 256, n_classi)),  # Immagine label
        )
    )
    trainset = trainset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valset = valset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    testset = testset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Ntest: ", len(test_camera_imgs_list_names))
    print("Nval: ", len(val_camera_imgs_list_names))
    print("Ntrain: ", len(train_camera_imgs_list_names))

    checkpoint_dir = "checkpoint/"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = checkpoint_dir + "modello.ckpt"

    checkpoint = ModelCheckpoint(checkpoint_path,
                                monitor='val_loss',  # Metrica da monitorare per la perdita
                                save_best_only=True,  # Salva solo i pesi migliori
                                mode='min',  # Minimizza la perdita
                                verbose=1) # Stampa un messaggio per mostrare che il salvataggio è avvenuto

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    def schedule_learning_rate(epoch, lr):
        if epoch < 10:
            return lr  # Tasso di apprendimento iniziale
        else:
            return lr * tf.math.exp(-0.1)  # Riduci il tasso di apprendimento ad ogni epoca successiva

    lr_scheduler = LearningRateScheduler(schedule_learning_rate)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001) #factror è il fattore di riduzione del tasso di apprendimento

    callbacks = [checkpoint, early_stopping, lr_scheduler, reduce_lr]

    class ResBlock(keras.Model):
        def __init__(self,num_hidden):
            super().__init__()
            self.conv1 = keras.layers.Conv2D(num_hidden, kernel_size = 3, padding = "same")
            self.bn1 = keras.layers.BatchNormalization()
            self.conv2 = keras.layers.Conv2D(num_hidden, kernel_size = 3, padding = "same")
            self.bn2 = keras.layers.BatchNormalization()
            
        def call(self, x):
            residui = x
            x = tf.nn.relu(self.bn1(self.conv1(x))) #il valore di input del "blocco" passa per "conv1", il suo output passa per bn1 (normalizza il batch, ossia li porta tra 0 e 1) e infine passa per la relu
            x = self.bn2(self.conv2(x))
            x += residui # questa è la skip connection (sommiamo l'input con l'output del blocco) # quindi l'insieme di filtri (batch x dimensione x dimensione x dim filtri)
            return tf.nn.relu(x)

        
    class ResNet(keras.Model):
        def __init__(self, n_blocchi, n_hidden):
            super().__init__()
            self.start_block = keras.Sequential([
                keras.layers.Conv2D(n_hidden, kernel_size = 3, padding = "same"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU()])
            self.back_bone = [ResBlock(n_hidden) for _ in range(n_blocchi)] # l'insieme di tutti i ResBlock
            self.fine = keras.Sequential([
                keras.layers.Conv2D(n_classi, kernel_size = 3, padding = "same"),
                keras.layers.BatchNormalization(),
                keras.layers.Softmax()]) # output della rete
        
        def call(self, x):
            x = self.start_block(x)
            for resblock in self.back_bone:
                x = resblock(x)
            risultato_rete = self.fine(x)
        
            return risultato_rete
        
    with strategy.scope():
        model_resnet = ResNet(n_blocchi = 15, n_hidden = 64)
        model_resnet.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=["accuracy"])
        model_resnet.build((None,256,256,3))
        model_resnet.summary()
        model_resnet.fit(x=trainset, epochs=10, validation_data=valset, initial_epoch=0)
        model_resnet.save_weights("resnet_py.keras")


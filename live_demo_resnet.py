import cv2

import numpy as np

import tensorflow as tf
from keras.losses import CategoricalCrossentropy

from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D


import keras
import cv2

#import albumentations as A

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
            keras.layers.Conv2D(55, kernel_size = 3, padding = "same"),
            keras.layers.BatchNormalization(),
            keras.layers.Softmax()]) # output della rete
    
    def call(self, x):
        x = self.start_block(x)
        for resblock in self.back_bone:
            x = resblock(x)
        risultato_rete = self.fine(x)
    
        return risultato_rete

    
model_resnet = ResNet(n_blocchi = 15, n_hidden = 64)
model_resnet.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=["accuracy"])
model_resnet.build((None,256,256,3))

model_resnet.load_weights("152_epoch_weights_256x256.keras")


#______________________________________________________________________________________________________________________
#Unet
class MyUnet(Model):
    def __init__(self, n_filters, n_classes):
        super().__init__()
        self.n_filters = n_filters
        self.down_conv = []
        self.up_conv = []

        self.max_pooling = MaxPool2D(data_format="channels_last")

        # down-section
        for n_filter in n_filters:
            self.down_conv.append(Conv2D(filters=n_filter, kernel_size=5, activation="relu", padding="same"))
            self.down_conv.append(Conv2D(filters=n_filter, kernel_size=3, activation="relu", padding="same"))

        # up-section
        for n_filter in reversed(n_filters):
            self.up_conv.append(Conv2DTranspose(filters=n_filter, kernel_size=2, strides=2, padding="same", activation="relu"))
            self.up_conv.append(Conv2D(filters=n_filter, kernel_size=3, padding="same", activation="relu"))

        self.last_conv = Conv2D(filters=n_classes, kernel_size=1, activation="softmax")

    def call(self, x):
        skips = []
        num_downsteps = len(self.down_conv) // 2

        # down part
        for i in range(num_downsteps):
            x = self.down_conv[2*i](x)
            x = self.down_conv[2*i+1](x)
            skips.append(x)
            x = self.max_pooling(x)

        # up part
        for i in range(num_downsteps):
            x = self.up_conv[2*i](x)
            skip = skips.pop()
            if x.shape[1:3] != skip.shape[1:3]:
                # Applica il padding personalizzato                                           #qui avvengono le magie di controllo della shape tra un layer e l'altro
                height_diff = skip.shape[1] - x.shape[1]                                      #qui avvengono le magie di controllo della shape tra un layer e l'altro
                width_diff = skip.shape[2] - x.shape[2]                                       #qui avvengono le magie di controllo della shape tra un layer e l'altro
                padding = tf.constant([[0, 0], [0, height_diff], [0, width_diff], [0, 0]])    #qui avvengono le magie di controllo della shape tra un layer e l'altro
                x = tf.pad(x, padding, "CONSTANT")                                            #qui avvengono le magie di controllo della shape tra un layer e l'altro
            x = x + skip                                                                      #qui avvengono le magie di controllo della shape tra un layer e l'altro
            x = self.up_conv[2*i+1](x)                                                        #qui avvengono le magie di controllo della shape tra un layer e l'altro


        return self.last_conv(x)
    
unet = MyUnet(n_filters=[32, 64, 128, 128, 256, 256], n_classes=55)
unet.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=["accuracy"])
unet.build((None,256,256,3))
unet.load_weights("unet_weights_256x256.keras")

# prendo un immagine da cv2 ed eseguo il resize a 256x256, passandola al modello, e visualizzo il risultato
#definisco la camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    pred = model_resnet.predict(frame)
    pred = np.squeeze(pred)
    pred = np.argmax(pred, axis=-1)
    
    # Converte l'output in un'immagine a colori
    color_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for class_idx in range(55):  # Sostituisci 55 con il numero di classi effettivo
        color_image[pred == class_idx] = (class_idx, class_idx, class_idx)  # Imposta il colore per ogni classe
        
    # Mostra l'immagine della previsione
    colormap = cv2.COLORMAP_JET

    # Applica la colormap
    colored_image = cv2.applyColorMap(color_image, colormap)
    # faccio un resize per poter visualizzare l'immagine
    colored_image = cv2.resize(colored_image, (1024, 1024))

    cv2.imshow("Segmentation Prediction", colored_image)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()
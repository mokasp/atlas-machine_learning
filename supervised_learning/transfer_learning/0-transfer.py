#!/usr/bin/env python3
""" module containing function that uses transfer learning to construct a
model that identifies images in the CIFAR-10 dataset """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ preprocesses data for training """
    inp = K.applications.inception_resnet_v2.preprocess_input(X)
    labels = K.utils.to_categorical(Y, 10)
    return inp, labels


def main():
    """ function that uses transfer learning to construct a model that
        identifies images in the CIFAR-10 dataset """
    train, test = K.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = train, test
    px_train, py_train = preprocess_data(x_train, y_train)
    px_test, py_test = preprocess_data(x_test, y_test)
    p_test = px_test, py_test

    data_gen = K.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True,
                                                        fill_mode='nearest')
    data_gen.fit(px_train)

    base_model = K.applications.InceptionResNetV2(include_top=False,
                                                  weights='imagenet')
    base_model.trainable = False
    inputs = K.Input(shape=(32, 32, 3))
    sc = K.layers.Lambda(lambda x:
                         K.backend.resize_images(
                             x, 229 // 32, 229 // 32,
                             data_format="channels_last",
                             interpolation="bilinear"))(inputs)
    x = base_model(sc, training=False)

    cbs = []
    save = K.callbacks.ModelCheckpoint('cifar10.h5', monitor='val_accuracy',
                                       save_best_only=True, mode='max')
    cbs.append(save)
    e_s = K.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    cbs.append(e_s)
    lrr = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                        patience=3, min_lr=0.00001)
    cbs.append(lrr)

    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.2)(x)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(1024)(x)
    x = K.layers.BatchNormalization(axis=1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(512)(x)
    x = K.layers.BatchNormalization(axis=1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(256)(x)
    x = K.layers.BatchNormalization(axis=1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x=data_gen.flow(px_train, py_train), epochs=50, batch_size=64,
              callbacks=cbs, shuffle=True, validation_data=p_test)


if __name__ == "__main__":
    main()

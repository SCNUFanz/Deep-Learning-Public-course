import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Dropout
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#使用http://course.gdou.com/上的catsdogs数据集
train_dir = 'catsdogs/train/'
validation_dir = 'catsdogs/valid/'
model_file_name = 'cat_dog_model.h5'


def init_model():
    model = models.Sequential()

    KERNEL_SIZE = (3, 3)

    model.add(layers.Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))  #二分类使用sigmoid函数

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  metrics=['accuracy'])

    return model


def fit(model):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        # steps_per_epoch=,
        epochs=30,
        validation_data=validation_generator,
        # validation_steps=,
    )

    model.save(model_file_name)

    plt.figure(figsize=(10, 6))

    #绘制训练时的准确率、损失率图
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')


    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'orange', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')


    plt.legend()
    plt.show()


def predict():
    model = load_model(model_file_name)
    print(model.summary())

    img_path = 'catsdogs/test/cats/b.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor / 255
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # 其形状为 (1, 150, 150, 3)
    plt.imshow(img_tensor[0])
    plt.show()

    result = model.predict(img_tensor)
    print(result)


# 画出count个预测结果和图像
def fig_predict_result(model, count):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        'catsdogs/valid/',
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')

    text_labels = []
    plt.figure(figsize=(30, 20))
    # 迭代器可以迭代很多条数据，但我这里只取第一个结果看看
    for batch, label in test_generator:
        pred = model.predict(batch)
        for i in range(count):
            true_reuslt = label[i]
            print(true_reuslt)
            if pred[i] > 0.5:
                text_labels.append('dog')
            else:
                text_labels.append('cat')
            # 4列，若干行的图
            plt.subplot(count / 4 + 1, 4, i + 1)
            plt.title('This is a ' + text_labels[i])
            imgplot = plt.imshow(batch[i])

        plt.show()
        break


if __name__ == '__main__':
    model = init_model()
    fit(model)

    # 利用训练好的模型预测结果。
    # 随机查看12个预测结果并预测结果
    model = load_model(model_file_name)
    fig_predict_result(model, 12)
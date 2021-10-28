import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers


#使用http://course.gdou.com/上的catsdogs数据集
#其中将train数据集中后500张猫和狗图片转移到test文件夹中


#训练数据集存储位置
base_dir = 'catsdogs'

#训练集 验证集 测试集
train_dir = base_dir + '/train'
validation_dir = base_dir + '/valid'
test_dir = base_dir + '/test'

#模型定义
conv_base = InceptionV3(weights='imagenet',include_top=False,input_shape=(299,299,3))

#层数冻结（查阅资料）
#InceptionV3的网络结构是由一块一块的Inception组成，每一块Inception包含了多种不同尺寸的卷积核、池化层、批标准化等组合。
#解冻最后两块Inception，通过观察网络结构可以定位到从第249层开始解冻
conv_base.trainable = True
for layer in conv_base.layers[:249]:
    layer.trainable = False
for layer in conv_base.layers[249:]:
    layer.trainable = True

#数据增强，调优部分

#正则化和Dropout
model = keras.models.Sequential([
    conv_base,
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256,activation='relu',kernel_regularizer=keras.regularizers.l2()),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1,activation='sigmoid')
])
#模型编译
model.compile(optimizer=keras.optimizers.SGD(lr=0.0001,momentum=0.9),loss='binary_crossentropy', metrics=['accuracy'])

#数据生成
img_width=299
img_height=299
img_channel=3
batch_size=32
epochs = 6

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#执行训练
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
    verbose=1)

#模型评估
score = model.evaluate(test_generator, steps=test_generator.n // batch_size)
print('测试准确率:{}, 测试loss值: {}'.format(score[1], score[0]))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Test Acc')
plt.title('Accauray')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train oss')
plt.plot(val_loss, label='Test Loss')
plt.title('Loss')
plt.legend()
plt.show()
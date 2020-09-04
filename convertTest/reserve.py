"""
モデルの保存と復元
"""
import os
import tensorflow as tf
from tensorflow import keras

#print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 短いシーケンシャルモデルを返す関数
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 基本的なモデルのインスタンスを作成
model = create_model()

# モデルの構造を表示
model.summary()

"""
訓練中にチェックポイントをさくせいする
"""
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 新しいコールバックを用いるようモデルを訓練
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 訓練にコールバックを渡す

# オプティマイザの状態保存についての警告が表示されるかもしれません。
# これらの警告は（このノートブックで発生する同様な警告を含めて）
# 古い用法を非推奨にするためのもので、無視して構いません。

# 訓練していない新しいモデルを作成
model = create_model()

loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

""" RESULT
32/32 - 0s - loss: 2.3782 - accuracy: 0.0960
Untrained model, accuracy:  9.60%
"""

#チェックポイントから重みをロード
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

""" RESULT
32/32 - 0s - loss: 0.4412 - accuracy: 0.8610
Restored model, accuracy: 86.10%
"""


"""
チェックポイントコールバック
"""
# ファイル名に(`str.format`を使って)エポック数を埋め込む
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 5エポックごとにモデルの重みを保存するコールバックを作成
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# 新しいモデルのインスタンスを作成
model = create_model()

# `checkpoint_path` フォーマットで重みを保存
model.save_weights(checkpoint_path.format(epoch=0))

# 新しいコールバックを使い、モデルを訓練
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# 新しいモデルのインスタンスを作成
model = create_model()

# 先ほど保存した重みを読み込み
model.load_weights(latest)

# モデルを再評価
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""RESULT
32/32 - 0s - loss: 0.4938 - accuracy: 0.8780
Restored model, accuracy: 87.80%
"""

"""
手動で重みを保存
"""
# 重みの保存
model.save_weights('./checkpoints/my_checkpoint')

# 新しいモデルのインスタンスを作成
model = create_model()

# 重みの復元
model.load_weights('./checkpoints/my_checkpoint')

# モデルの評価
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""RESULT
32/32 - 0s - loss: 0.4948 - accuracy: 0.8740
Restored model, accuracy: 87.40%
"""

"""
モデル全体の保存
"""
# 新しいモデルのインスタンスを作成して訓練
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# モデル全体を SavedModel として保存
#!mkdir -p saved_model
model.save('saved_model/my_model')

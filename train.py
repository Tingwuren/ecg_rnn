import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import build_rnn_model

# 显示可用的 GPU 设备
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 设置随机种子以保证结果复现性
import random
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 加载预处理后的数据（假设已经处理并保存在 npy 文件中）
def load_data():
    # 加载处理后的 ECG 数据，假设每个样本为 200 个时间步
    data = np.load("ecg_filtered_100.npy")
    data_segments = data.reshape(-1, 200, 1)  # 假设每个样本为200个时间步
    labels = np.zeros(len(data_segments))  # 仅示例，真实任务中需要根据注释生成标签
    return data_segments, labels

# 加载数据
X, y = load_data()

# 划分数据集（70% 训练，15% 验证，15% 测试）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 构建模型
model = build_rnn_model(timesteps=200, num_features=1, num_classes=2)

# 配置回调函数（早停和模型检查点）
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# 训练模型
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping, checkpoint])

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# 可视化训练过程
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()
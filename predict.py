import numpy as np
import wfdb
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import preprocess_data  # 引用预处理函数

# 加载训练好的模型
model = load_model('best_model.h5')

# 使用模型进行预测
def predict_ecg(record_id='101', window_size=200, start_time=0, end_time=60):
    """
    预测给定 ECG 数据集的结果，限制显示时间范围在 start_time 和 end_time 之间
    """
    # 预处理数据
    ecg_data = preprocess_data(record_id, window_size)

    # 获取采样率
    fs = 360  # 采样率，单位：Hz
    # 限制数据为指定时间段内的样本
    min_samples = int(start_time * fs)
    max_samples = int(end_time * fs)

    # 截取对应时间段的数据
    ecg_data = ecg_data[min_samples // window_size:max_samples // window_size]

    # 使用模型进行预测
    predictions = model.predict(ecg_data)

    # 获取预测结果（假设为二分类：0 正常，1 异常）
    predicted_classes = np.argmax(predictions, axis=1)

    return ecg_data, predicted_classes

# 可视化 ECG 信号与预测结果
def plot_ecg_with_time(record_id='101', window_size=200, start_time=0, end_time=60):
    """
    使用时间轴绘制 ECG 信号，允许用户查看任意时间段的 ECG 信号
    """
    # 读取整个 ECG 信号
    record = wfdb.rdrecord(f'./data/mitdb/{record_id}')
    ecg_signal = record.p_signal[:, 0]  # 使用第一个通道的 ECG 信号

    # 获取采样率
    fs = record.fs  # 采样频率，单位为 Hz

    # 创建时间轴（单位：秒）
    time_axis = np.arange(len(ecg_signal)) / fs  # 每个数据点对应的时间

    # 根据给定的时间范围限制数据
    min_samples = int(start_time * fs)
    max_samples = int(end_time * fs)
    ecg_signal = ecg_signal[min_samples:max_samples]
    time_axis = time_axis[min_samples:max_samples]

    # 获取预测结果
    ecg_data, predicted_classes = predict_ecg(record_id, window_size, start_time, end_time)

    # 设置图像大小（宽度增加，以便更好地查看信号）
    plt.figure(figsize=(15, 6))  # 增加图像的宽度

    # 绘制信号
    plt.plot(time_axis, ecg_signal, label="ECG Signal", color='blue')

    # 标注预测为异常的部分（假设预测结果为 1 的部分是异常）
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == 1:  # 异常
            # 标记异常部分，红色标记
            plt.axvspan(i * window_size / fs + start_time, (i + 1) * window_size / fs + start_time, color='red', alpha=0.5)

    # 优化横坐标显示密度
    # 使用 np.linspace 生成合理的刻度
    xticks = np.linspace(start_time, end_time, num=11)  # 生成从 start_time 到 end_time 的 11 个点
    plt.xticks(xticks, rotation=45)  # 设置 x 轴刻度并旋转标签

    # 标题和标签
    plt.title(f"ECG Signal with Predicted Anomalies (Record {record_id})")
    plt.xlabel("Time (s)")  # 横坐标单位为秒
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # 设置横坐标的范围
    plt.xlim([start_time, end_time])

    # 调整布局，避免图像显示过于拥挤
    plt.tight_layout()

    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 可视化 ECG 信号，并标注预测为异常的部分
    plot_ecg_with_time('101', window_size=200, start_time=30, end_time=60)  # 查看从 30 秒到 60 秒的信号
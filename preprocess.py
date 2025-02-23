import wfdb
import numpy as np
import scipy.signal as signal

# 读取 ECG 数据
def load_record(record_id='100'):
    """
    加载ECG数据记录（从 PhysioNet 数据库）
    """
    record = wfdb.rdrecord(f'./data/mitdb/{record_id}')
    annotation = wfdb.rdann(f'./data/mitdb/{record_id}', 'atr')  # 获取标注数据
    return record, annotation

# 带通滤波函数
def bandpass_filter(signal_data, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """
    对 ECG 信号进行带通滤波，去除噪声
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)

# 数据分割函数
def preprocess_data(record_id='100', window_size=200):
    """
    读取并处理 ECG 信号，将其分割成小片段
    """
    record, annotation = load_record(record_id)

    # 提取 ECG 信号
    ecg_signal = record.p_signal[:, 0]  # 使用第一个通道的数据

    # 对 ECG 信号进行滤波
    filtered_ecg = bandpass_filter(ecg_signal)

    # 分割信号
    samples = []
    for i in range(0, len(filtered_ecg) - window_size, window_size):
        samples.append(filtered_ecg[i:i + window_size])

    return np.array(samples)

if __name__ == "__main__":
    preprocess_data('100')  # 处理 record 100
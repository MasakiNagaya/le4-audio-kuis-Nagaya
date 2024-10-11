#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 正弦波を生成し，音声ファイルとして出力する
#

import sys
import math
import numpy as np
import scipy.io.wavfile

# 正弦波を生成する関数
# sampling_rate ... サンプリングレート
# frequency ... 生成する正弦波の周波数
# duration ... 生成する正弦波の時間的長さ
def generate_sinusoid(sampling_rate, frequency, duration):
	sampling_interval = 1.0 / sampling_rate
	t = np.arange(sampling_rate * duration) * sampling_interval
	waveform = np.sin(2.0 * math.pi * frequency * t)
	return waveform

# サンプリングレート
sampling_rate = 16000.0

# 生成する正弦波の周波数（Hz）
frequency = 440.0

# 生成する正弦波の時間的長さ
duration = 1.0 # seconds

# 正弦波を生成する
c5 = generate_sinusoid(sampling_rate, 523.25, duration) #C5
e5 = generate_sinusoid(sampling_rate, 659.26, duration) #E5
g5 = generate_sinusoid(sampling_rate, 783.99, duration) #G5

ef5 = generate_sinusoid(sampling_rate, 622.25, duration) #E♭
f5 = generate_sinusoid(sampling_rate, 698.46, duration) #f5
a5 = generate_sinusoid(sampling_rate, 880.00, duration) #a5
b4 = generate_sinusoid(sampling_rate, 493.88, duration) #G5

# 最大値を0.9にする
waveform1 = (c5 + e5 + g5)*3/10
waveform2 = (c5 + f5 + a5)*3/10
waveform3 = (b4 + f5 + g5)*3/10
waveform4 = (c5 + ef5 + g5)*3/10
waveform5 = (c5 + e5 + g5)*3/10

waveform = np.append(waveform1,waveform2)
waveform = np.append(waveform,waveform3)
waveform = np.append(waveform,waveform4)
waveform = np.append(waveform,waveform5)

# 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
waveform = (waveform * 32768.0). astype('int16')

# 音声ファイルとして出力する
filename = 'rec/sinuoid_ex14.wav'
scipy.io.wavfile.write(filename , int(sampling_rate), waveform)
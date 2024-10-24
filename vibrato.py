#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 正弦波の乗算によるボイスチェンジャ
#

import sys
import math
import numpy as np
import scipy.io.wavfile
import librosa

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
SR = 16000

name = "aiueo_02_test"
# 音声ファイルの読み込み
x, _ = librosa.load('rec/'+name+'.wav', sr=SR)

f_s = 1
R = 1.5
D = 10

# 生成する正弦波の周波数（Hz）
frequency = 2 * np.pi * R / f_s

# 生成する正弦波の時間的長さ
duration = len(x)

# 正弦波を生成する
sin_wave = generate_sinusoid(SR, frequency, duration/SR)

# 最大値を0.9にする
tau = D * sin_wave

# 音声波形を時間軸上で前後で揺らす
x_changed = np.zeros_like(x)
for t in np.arange(len(x)-1):
	if(t-int(tau[t]) > 0 and t-int(tau[t]) < len(x)-2):
		x_changed[t] = x[t-int(tau[t])]
	else:
		x_changed[t] = x[t]

# 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
x_changed = (x_changed * 32768.0). astype('int16')

# 音声ファイルとして出力する
filename = 'rec/vibrato' + name + 'D='+str(D)+'R='+str(R)+'.wav'
scipy.io.wavfile.write(filename , int(SR), x_changed)
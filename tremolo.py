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

name = "shs-test"
# 音声ファイルの読み込み
x, _ = librosa.load('rec/'+name+'.wav', sr=SR)

f_s = 1
R = 1
D = 0.2

# 生成する正弦波の周波数（Hz）
frequency = 2 * np.pi * R / f_s

# 生成する正弦波の時間的長さ
duration = len(x)

# 正弦波を生成する
sin_wave = generate_sinusoid(SR, frequency, duration/SR)

# 最大値を0.9にする
sin_wave = ((1 + D * sin_wave)/(1+D)) * 0.9 

# 元の音声と正弦波を重ね合わせる
x_changed = x * sin_wave

# 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
x_changed = (x_changed * 32768.0). astype('int16')

# 音声ファイルとして出力する
filename = 'rec/tremolo' + name + 'D0.2R1.wav'
scipy.io.wavfile.write(filename , int(SR), x_changed)
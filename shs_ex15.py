#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，スペクトログラムを計算して図示する．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math

# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ変換（notenumber.pyより）
def hz2nn(frequency):
	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69


name = "momotaro10"

# サンプリングレート
SR = 16000
# 音声ファイルの読み込み
x, _ = librosa.load('rec/' + name + '.wav', sr=SR)

#
# 波形を画像に表示・保存
#

# 画像として保存するための設定
# 画像サイズを 1000 x 400 に設定
fig = plt.figure(figsize=(10, 4))

# 波形を描画
plt.plot(np.arange(len(x))/SR,x)				# 描画データを追加
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.xticks(np.linspace(0,len(x)/SR,15))
plt.grid()
plt.show()										# 表示

# 画像ファイルに保存
fig.savefig('picture/' + name + '_waveform_ex15.png')

# フレームサイズ
size_frame = 4096	# 2のべき乗
# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)
# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []
# sub-harmonic summation (SHS)
# 全高調波成分とは、交流波形を各周波数成分に分解した際に、
# 基本波の2倍以上の周波数成分を二乗して総和し、平方根を取ったものです。
shs_list = []
note_list = np.arange(0,80)
spec_power = np.full_like(note_list, -99999.9)
start = int((nn2hz(36) * (size_frame) /SR))
end = int((nn2hz(60) * (size_frame) /SR))

print(start, end)
print(np.arange(start, end))

for i in np.arange(0, len(x)-size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    x_frame = x[idx : idx+size_frame]
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec[:int(size_frame/16)]) # 2048/8 = 256

    for index in np.arange(start, end): # len(fft_spec)=2049 : 8000Hz  
        spec_power[index] = 0
        for k in np.arange(1,5):
            spec_power[index] += (fft_log_abs_spec[index * k] + fft_log_abs_spec[index * k + 1]/4 + fft_log_abs_spec[index * k - 1]/4) / (k)
    # shs_list.append(nn2hz(np.argmax(spec_power) + 36))
    shs = hz2nn(np.argmax(spec_power) * (SR/size_frame))
    shs_list.append(shs)

    # for j in np.arange(0, len(note_list)): # len(fft_spec)=2049 : 8000Hz  
    #     for k in np.arange(1,5):
    #         spec_index = round((nn2hz(note_list[j]) * k * (size_frame) /SR))
    #         # print(spec_index)
    #         spec_power[j] += fft_log_abs_spec[spec_index]
    # # shs_list.append(nn2hz(np.argmax(spec_power) + 36))
    # shs_list.append(np.argmax(spec_power) + 36)

print(len(spectrogram))
print(len(shs_list))
print(len(fft_spec))
print(shs_list)
print(spec_power)

#
# 音高を画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
plt.plot(np.linspace(0,len(x)/SR,len(shs_list)),shs_list)				# 描画データを追加
# plt.xticks(np.linspace(0,len(x)/SR,15))
plt.grid()
plt.show()

# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
fig.savefig('picture/' + name + 'shs_ex16.png')

#
# スペクトログラムを画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
plt.imshow(
	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, len(x)/SR, 0, SR/(2*16)],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.show()

# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
fig.savefig('picture/' + name + 'spectrogram_ex16.png')



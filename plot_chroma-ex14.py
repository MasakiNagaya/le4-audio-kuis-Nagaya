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

# 周波数からノートナンバーへ変換（notenumber.pyより）
def hz2nn(frequency):
	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

def chroma_vector(spectrum, frequencies):
	cv = np.zeros(12)
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn % 12] += abs(s)
	return cv

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('rec/sinuoid_ex14.wav', sr=SR)

# フレームサイズ
size_frame = 4096		# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []
# クロマグラフを保存するlist
chromagram = []
# 和音を保存するリスト
chord = []

a_root = 1
a_3 = 0.5
a_5 = 0.8

for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]
	
	# スペクトログラム
	fft_spec = np.fft.rfft(x_frame * hamming_window)
	fft_log_abs_spec = np.log(np.abs(fft_spec))
	spectrogram.append(fft_log_abs_spec[:256]) # /8 = 256

	# クロマグラフ
	frequencies = np.linspace((SR/2)/len(fft_spec), SR/2, len(fft_spec))
	chroma = chroma_vector(fft_spec,frequencies)
	chromagram.append(chroma)

	#和音
	chroma = np.append(chroma,chroma)
	l=[]
	for i in range(0,12):
		l.append(a_root*chroma[i] + a_3*chroma[i+4] + a_5*chroma[i+7])
		l.append(a_root*chroma[i] + a_3*chroma[i+3] + a_5*chroma[i+7])
	chord.append(np.argmax(np.array(l)))

#
# スペクトログラムを画像に表示・保存
#

# # 画像として保存するための設定
fig = plt.figure()

# # スペクトログラムを描画
# plt.xlabel('time(s)')				# x軸のラベルを設定
# plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
# plt.imshow(
# 	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転置して上下反転
# 	extent=[0, len(x)/16000, 0, SR/(2*8)],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
# 	aspect='auto',
# 	interpolation='nearest'
# )
# plt.show()
# fig.savefig('picture/plot-spectogram-ex14.png')

# クロマグラムを描画
plt.xlabel('time(s)')				# x軸のラベルを設定
plt.ylabel('chroma')				# y軸のラベルを設定
plt.imshow(
	np.flipud(np.array(chromagram).T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, len(x)/16000, 0, 24],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.plot(np.linspace(0, len(x)/16000,len(chord)),chord, color="white")
plt.yticks(range(0,24),
            ['C','Cm','C#','C#m','D','Dm','D#','D#m','E','Em','F','Fm','F#','F#m','G','Gm','G#','G#m','A','Am','A#','A#m','B','Bm'])

plt.show()
fig.savefig('picture/plot-chroma-ex14.png')

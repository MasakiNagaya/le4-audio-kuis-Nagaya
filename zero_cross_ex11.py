#ex11 ゼロ交差数 zero crossing (rate)

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	if a[index-1] < a[index] and a[index+1] < a[index]:
		return True
	else: 
		return False
	
# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
	zc = 0
	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1
	return zc

# 音声波形データを受け取り，ゼロ交差数を計算する関数
# 簡潔版
def zero_cross_short(waveform):	
	d = np.array(waveform)
	return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('rec/ex11.wav', sr=SR)

# フレームサイズ
size_frame = 4096		# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []
# 基本周波数を保存するlist
fundamental_frequency = []
# zero cross
zero_cross_list = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	idx = int(i)
	x_frame = x[idx : idx+size_frame]

	# 自己相関
	autocorr = np.correlate(x_frame, x_frame, 'full')
	autocorr = autocorr [len (autocorr ) // 2 : ]
	# fft_spec = np.fft.rfft(x_frame)
	# autocorr = np.fft.irfft(fft_spec ** 2)
	peakindices = [i for i in range (len (autocorr)-1) if is_peak (autocorr, i)]
	peakindices = [i for i in peakindices if i != 0]
	if peakindices!=[]:
		max_peak_index = max(peakindices , key=lambda index: autocorr [index])
		tau = max_peak_index/SR

		# zero_cross
		zero_cross_rate = zero_cross(x_frame) / (size_frame / SR)
		zero_cross_list.append(zero_cross_rate)

		if zero_cross_rate > (1/tau)*1 and zero_cross_rate < (1/tau)*11:
			if (1/tau)<400:fundamental_frequency.append(1/tau)
		else:
			fundamental_frequency.append(0)

		# スペクトログラム
		fft_spec = np.fft.rfft(x_frame * hamming_window)
		fft_log_abs_spec = np.log(np.abs(fft_spec))
		spectrogram.append(fft_log_abs_spec[:256]) # /8 = 256

#
# スペクトログラムを画像に表示・保存
#

# # 画像として保存するための設定
# fig = plt.figure()

# # zero_crossを描画
# plt.xlabel('time(s)')					# x軸のラベルを設定
# plt.ylabel('frequency[Hz]')		# y軸のラベルを設定
# # plt.ylim(0,500)
# plt.plot(np.linspace(0,(len(x)-size_frame)/16000, len(zero_cross_list)), zero_cross_list)
# plt.show()

# # 画像ファイルに保存
# fig.savefig('picture/plot_zero_cross.png')

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
plt.ylim(0, SR/(2*8))
plt.imshow(
	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, (len(x)-size_frame)/16000, 0, SR/(2*8)],		# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.plot(np.linspace(0,(len(x)-size_frame)/16000, len(fundamental_frequency)), fundamental_frequency)
plt.show()

# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
fig.savefig('picture/plot-spectogram_zero-cross-ex11_6~11_400.png')
#ex12

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	# 1. 振幅スペクトルの対数をとる．
	log_spectrum = np.log(amplitude_spectrum)
	# 2. 対数振幅スペクトルをフーリエ変換する．
	cepstrum = np.fft.rfft(log_spectrum)
	return cepstrum[0:20]

# サンプリングレート
SR = 16000
# フレームサイズ
size_frame = 2048		# 2のべき乗
# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)
# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

#					#
#	 学習フェーズ	 #
#					#

# mu_hatを保存するlist
mu_hat_list = []
# sigma_hatを保存するlist
sigma_hat_list = []

for name in ['a','i','u','e','o']:

	# 音声ファイルの読み込み
	x, _ = librosa.load('rec/'+name+'_train.wav', sr=SR)

	# ケプストラムを保存するlist
	cepstrum_spec_list = []

	for i in np.arange(0, len(x)-size_frame, size_shift):
		
		idx = int(i)
		x_frame = x[idx : idx+size_frame]

		# ケプストラム
		fft_spec = np.fft.rfft(x_frame * hamming_window)
		cepstrum_spec = cepstrum(fft_spec)
		cepstrum_spec_list.append(np.abs(cepstrum_spec))

	mu_hat = np.mean(cepstrum_spec_list, axis=0)
	sigma_hat = np.var(cepstrum_spec_list, axis=0)

	mu_hat_list.append(mu_hat)
	sigma_hat_list.append(sigma_hat)

# print(mu_hat_list[:][:13])
# print(sigma_hat_list[:][:13])

#					#
#	 音素の認識		 #
#					#

def model(cepstrum_spec_list):
	# 対数尤度のlist
	log_L_list = []
	for i in np.arange(0,5):
		l = 0
		for n in np.arange(0,np.shape(cepstrum_spec_list)[0]):
			l += np.sum(np.log(sigma_hat_list[i])/2 + ((cepstrum_spec_list[n]-mu_hat_list[i])**2)/(2*sigma_hat_list[i]))
		log_L_list.append(-l)
	# print(log_L_list)
	return np.argmax(log_L_list)


# 音声ファイルの読み込み
# x, _ = librosa.load('rec/a_train.wav', sr=SR)
# x, _ = librosa.load('rec/aiueo_01_train.wav', sr=SR)
x, _ = librosa.load('rec/aiueo_02_test.wav', sr=SR)

# スペクトログラムを保存するlist
spectrogram = []
#結果のlist
aiueo_list = []
#counter
counter = 0
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	counter += 1

	idx = int(i)
	x_frame = x[idx : idx+size_frame]

	# ケプストラム
	fft_spec = np.fft.rfft(x_frame * hamming_window)
	cepstrum_spec = cepstrum(fft_spec)
	cepstrum_spec_list.append(np.abs(cepstrum_spec))

	#spectrogram
	fft_log_abs_spec = np.log(np.abs(fft_spec))
	spectrogram.append(fft_log_abs_spec[:256]) # /8 = 256

	r = counter-1
	l = np.max([r-30, 0])
	aiueo = model(cepstrum_spec_list[l:r])
	aiueo_list.append(aiueo)

print(len(aiueo_list))
print(len(spectrogram))

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
	extent=[0, len(x)/SR, 0, SR/(2*8)],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
aiueo_list = np.array(aiueo_list) * 100 + 100
plt.plot(np.linspace(0, (len(x)-size_frame)/SR, len(aiueo_list)), aiueo_list, color="red")
plt.show()

# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
fig.savefig('picture/plot-spectogram-ex12test.png')


# # 画像として保存するための設定
# fig = plt.figure()

# # スペクトログラムを描画
# plt.xlabel('sample')					# x軸のラベルを設定
# plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
# plt.plot(aiueo_list)
# plt.show()

# # 【補足】
# # 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# # 画像ファイルに保存
# fig.savefig('picture/plot-aiueo-ex12.png')
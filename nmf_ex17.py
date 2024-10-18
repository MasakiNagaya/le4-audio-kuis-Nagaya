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

name = "nmf_piano_sample"

# サンプリングレート
SR = 16000
# 音声ファイルの読み込み
x, _ = librosa.load('rec/' + name + '.wav', sr=SR)

# フレームサイズ
size_frame = 4096	# 2のべき乗
# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)
# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []

for i in np.arange(0, len(x)-size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    x_frame = x[idx : idx+size_frame]
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec) + 1)
    # spectrogram.append(fft_log_abs_spec[:int(size_frame/16)]) # 2048/8 = 256
    spectrogram.append(fft_log_abs_spec)

Y = np.array(spectrogram).T # f * t
f,t = np.shape(Y)
k = 3
# H, U を作成 Y = H * U
np.random.seed(91)
H = np.abs(np.random.uniform(low=0, high=1, size=(f,k)))
U = np.abs(np.random.uniform(low=0, high=1, size=(k,t)))

# 更新式
for i in range(1000):
    H = H * (Y @ U.T) / (U @ (H @ U).T).T
    U = U * (Y.T @ H).T / (H.T @ (H @ U))
    NMF = H @ U
    D = np.mean(np.square(Y-NMF))
    print(D)
    if np.abs(D) < 0.01:
        break

spectrograms = []

# 位相復元
for i in range(k):
    HH = np.zeros_like(H)
    HH.T[i] = H.T[i]
    spectrograms.append(HH @ U)

#
# スペクトログラムを画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('frequency(Hz)')					# x軸のラベルを設定
plt.ylabel('num')		# y軸のラベルを設定
plt.imshow(
	np.flipud(H.T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, SR/2, 0,k],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.show()

# 画像ファイルに保存
fig.savefig('picture/' + name + 'H_ex17_k=' + str(k) + '.png')

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('num')		# y軸のラベルを設定
plt.imshow(
	np.flipud(U),		# 画像とみなすために，データを転置して上下反転
	extent=[0, len(x)/SR, 0,k],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.show()

# 画像ファイルに保存
fig.savefig('picture/' + name + 'U_ex17_k=' + str(k) + '.png')


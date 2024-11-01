# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込みスペクトログラム上に音量を表示する
#

# ライブラリの読み込み
import librosa
import numpy as np
import matplotlib.pyplot as plt

size_frame = 4096			# フレームサイズ
SR = 16000					# サンプリングレート
size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)

# 音声ファイルを読み込む
x, _ = librosa.load('rec/aiueo_02_test.wav', sr=SR)

x1, _ = librosa.load('rec/vibratoaiueo_02_testD=5R=1.wav', sr=SR)

x2, _ = librosa.load('rec/vibratoaiueo_02_testD=10R=0.9.wav', sr=SR)
x3, _ = librosa.load('rec/vibratoaiueo_02_testD=10R=1.5.wav', sr=SR)
x4, _ = librosa.load('rec/vibratoaiueo_02_testD=10R=1.wav', sr=SR)
x5, _ = librosa.load('rec/vibratoaiueo_02_testD=10R=2.wav', sr=SR)
x6, _ = librosa.load('rec/vibratoaiueo_02_testD=10R=3.wav', sr=SR)
x7, _ = librosa.load('rec/vibratoaiueo_02_testD=50R=0.2.wav', sr=SR)
x8, _ = librosa.load('rec/vibratoaiueo_02_testD=100R=0.2.wav', sr=SR)
x9, _ = librosa.load('rec/vibratoaiueo_02_testD=100R=1.wav', sr=SR)
xA, _ = librosa.load('rec/vibratoaiueo_02_testD=300R=0.2.wav', sr=SR)
xB, _ = librosa.load('rec/vibratoaiueo_02_testD=500R=0.2.wav', sr=SR)

X = [x,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB]
labels = ["default", "D=5R=1", "D=10R=0.9", "D=10R=1.5", "D=10R=1", "D=10R=2", "D=10R=3", "D=50R=0.2", "D=100R=0.2", "D=100R=1", "D=300R=0.2", "D=500R=0.2"]

# ファイルサイズ（秒）
duration = len(x) / SR
size_frame = 4096
hamming_window = np.hamming(size_frame)
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

spectrograms = []

for j in np.arange(len(X)):

    spectrogram = []

    for i in np.arange(0, len(X[j])-size_frame, size_shift):
        idx = int(i)
        x_frame = X[j][idx : idx+size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec[:256]) # /8 = 256

    spectrograms.append(spectrogram)

#
# スペクトログラムを画像に表示・保存
#
fig, axes = plt.subplots(4, 4, tight_layout=True,figsize=(30, 30))

# 波形を描画

axes[0,0].set_xlabel("time[sec]")
axes[0,0].set_ylabel("frequency [Hz]")
axes[0,0].imshow(
    np.flipud(np.array(spectrograms[0]).T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, len(x)/SR, 0, SR/(2*8)],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
axes[0,0].set_title("default")

for i in np.arange(1,len(X)):
    axes[i%4, int(i/4)+1].set_xlabel("time[sec]")
    axes[i%4, int(i/4)+1].set_ylabel("frequency [Hz]")
    axes[i%4, int(i/4)+1].imshow(
        np.flipud(np.array(spectrograms[i]).T),		# 画像とみなすために，データを転置して上下反転
        extent=[0, len(X[i])/SR, 0, SR/(2*8)],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
        aspect='auto',
        interpolation='nearest'
    )
    axes[i%4, int(i/4)+1].set_title(labels[i])
plt.show()		
fig.savefig('picture/aiueo_02_test_spectrogram_vibrato_extra6.png')
#
# 波形を画像に表示・保存
#

# 画像として保存するための設定
# 画像サイズを 1000 x 400 に設定
# fig = plt.figure(figsize=(10, 4))
fig, axes = plt.subplots(4, 4, tight_layout=True,figsize=(30, 12))

# 波形を描画

axes[0,0].set_xlabel("time[sec]")
# axes[0,0].set_xticks(np.linspace(0,len(x)/SR),5)
axes[0,0].plot(np.arange(len(x))/SR,x,label="default")
axes[0,0].grid()
axes[0,0].legend()

for i in np.arange(1,len(X)):
    axes[i%4, int(i/4)+1].set_xlabel("time[sec]")
    # axes[i%4, int(i/4)].set_xticks(np.linspace(0,len(X[i])/SR),5)
    axes[i%4, int(i/4)+1].plot(np.arange(len(X[i]))/SR,X[i],label=labels[i])
    axes[i%4, int(i/4)+1].grid()
    axes[i%4, int(i/4)+1].legend()
plt.show()										# 表示

# 画像ファイルに保存
fig.savefig('picture/aiueo_02_test_waveform_vibrato_extra6.png')





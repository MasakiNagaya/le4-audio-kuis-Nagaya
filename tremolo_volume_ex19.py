#
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
x, _ = librosa.load('rec/shs-test.wav', sr=SR)
x1, _ = librosa.load('rec/tremoloshs-testD1R1.wav', sr=SR)
x2, _ = librosa.load('rec/tremoloshs-testD1R.2.wav', sr=SR)
x3, _ = librosa.load('rec/tremoloshs-testD1R3.wav', sr=SR)
x4, _ = librosa.load('rec/tremoloshs-testD0.2R1.wav', sr=SR)
x5, _ = librosa.load('rec/tremoloshs-testD3R1.wav', sr=SR)

# ファイルサイズ（秒）
duration = len(x) / SR

# ハミング窓
hamming_window = np.hamming(size_frame)

volume = []			# 音量を保存するlist
volume1 = []
volume2 = []
volume3 = []
volume4 = []
volume5 = []

# フレーム毎に処理
for i in np.arange(0, len(x)-size_frame, size_shift):
	
    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換

    x_frame = x[idx : idx+size_frame]
    # 音量
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume.append(vol)

    x_frame = x1[idx : idx+size_frame]
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume1.append(vol)

    x_frame = x2[idx : idx+size_frame]
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume2.append(vol)

    x_frame = x3[idx : idx+size_frame]
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume3.append(vol)

    x_frame = x4[idx : idx+size_frame]
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume4.append(vol)

    x_frame = x5[idx : idx+size_frame]
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume5.append(vol)

# 画像として保存するための設定
fig = plt.figure()

# まずはスペクトログラムを描画
ax1 = fig.add_subplot(111)
ax1.set_xlabel('time [sec]')
ax1.set_ylabel('volume [dB]')
x_data = np.linspace(0, duration, len(volume))
ax1.set_xlim([0,3])
ax1.set_ylim([-80,-28])
ax1.plot(x_data, volume, label="default")
ax1.plot(x_data, volume1, label="D1R1")
ax1.plot(x_data, volume2, label="D1R0.2")
ax1.plot(x_data, volume3, label="D1R3")
ax1.plot(x_data, volume4, label="D0.2R1")
ax1.plot(x_data, volume5, label="D3R1")
plt.grid()
plt.legend()

plt.show()
fig.savefig('picture/plot-spectogram-volume_tremolo_ex19.png')


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

size_frame = 4096			# フレームサイズ
SR = 16000					# サンプリングレート
size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)

# 音声ファイルを読み込む
x, _ = librosa.load('rec/aiueo2.wav', sr=SR)

# ファイルサイズ（秒）
duration = len(x) / SR
# ハミング窓
hamming_window = np.hamming(size_frame)

spectrogram = []	# スペクトログラムを保存するlist
spectrogram_all = []
volume = []			# 音量を保存するlist
fundamental_frequency = []  # 基本周波数を保存するlist
zero_cross_list = []    # zero cross

# フレーム毎に処理
for i in np.arange(0, len(x)-size_frame, size_shift):
	
    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    x_frame = x[idx : idx+size_frame]

    # 自己相関
    autocorr = np.correlate(x_frame, x_frame, 'full')
    autocorr = autocorr [len (autocorr ) // 2 : ]
    peakindices = [i for i in range (len (autocorr)-1) if is_peak (autocorr, i)]
    peakindices = [i for i in peakindices if i != 0]
    if peakindices!=[]:
        max_peak_index = max(peakindices , key=lambda index: autocorr [index])
        tau = max_peak_index/SR

        # zero_cross
        zero_cross_rate = zero_cross(x_frame) / (size_frame / SR)
        zero_cross_list.append(zero_cross_rate)

        if zero_cross_rate > (1/tau)*3 and zero_cross_rate < (1/tau)*11:
            if (1/tau)<400:fundamental_frequency.append(1/tau)
        else:
            fundamental_frequency.append(0)

        # スペクトログラム
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec[:256]) # /8 = 256
        
        spectrogram_all.append(fft_log_abs_spec)

        # 音量
        vol = 20 * np.log10(np.mean(x_frame ** 2))
        volume.append(vol)

# 画像として保存するための設定
fig = plt.figure()

# まずはスペクトログラムを描画
ax1 = fig.add_subplot(111)
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')
ax1.imshow(
	np.flipud(np.array(spectrogram).T),
	extent=[0, duration, 0, 8000/8],
	aspect='auto',
	interpolation='nearest'
)
ax1.plot(np.linspace(0,(len(x)-size_frame)/16000, len(fundamental_frequency)), fundamental_frequency, color='red')

# 続いて右側のy軸を追加して，音量を重ねて描画
ax2 = ax1.twinx()
ax2.set_ylabel('volume [dB]')
x_data = np.linspace(0, duration, len(volume))
ax2.plot(x_data, volume, c='y')

plt.show()
fig.savefig('picture/plot-spectogram-volume_aiueo2.png')


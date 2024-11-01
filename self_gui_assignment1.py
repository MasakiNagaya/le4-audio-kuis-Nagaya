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
import tkinter

# MatplotlibをTkinterで使用するために必要
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	# 1. 振幅スペクトルの対数をとる．
	log_spectrum = np.log(amplitude_spectrum)
	# 2. 対数振幅スペクトルをフーリエ変換する．
	cepstrum = np.fft.rfft(log_spectrum)
	return cepstrum[0:20]

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

size_frame = 2048			# フレームサイズ
SR = 16000					# サンプリングレート
size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)
# ハミング窓
hamming_window = np.hamming(size_frame)


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




# 音声ファイルを読み込む
# x, _ = librosa.load('rec/aiueo2.wav', sr=SR)
x, _ = librosa.load('rec/aiueo_02_test.wav', sr=SR)

# ファイルサイズ（秒）
duration = len(x) / SR

spectrogram = []	        # スペクトログラムを保存するlist
spectrogram_all = []
volume = []			        # 音量を保存するlist
fundamental_frequency = []  # 基本周波数を保存するlist
zero_cross_list = []        # zero cross
aiueo_list = []             #結果のlist
counter = 0                 #counter


# フレーム毎に処理
for i in np.arange(0, len(x)-size_frame, size_shift):
    
    counter += 1

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
        
        # ケプストラム
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        cepstrum_spec = cepstrum(fft_spec)
        cepstrum_spec_list.append(np.abs(cepstrum_spec))

        #aiueo
        r = counter-1
        l = np.max([r-30, 0])
        aiueo = model(cepstrum_spec_list[l:r])
        aiueo_list.append(aiueo)

        # スペクトログラム
        # fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec[:128]) # 2048/8 = 256
        
        spectrogram_all.append(fft_log_abs_spec)

        # 音量
        vol = 20 * np.log10(np.mean(x_frame ** 2))
        volume.append(vol)

print(len(fft_log_abs_spec))

# Tkinterを初期化
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-MASAKI NAGAYA-ASSIGNMENT1")

# Tkinterのウィジェットを階層的に管理するためにFrameを使用
# frame1 ... スペクトログラムを表示
# frame2 ... Scale（スライドバー）とスペクトルを表示
frame1 = tkinter.Frame(root)
frame2 = tkinter.Frame(root)
frame1.pack(side="left")
frame2.pack(side="left")

# まずはスペクトログラムを描画
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame1)	# masterに対象とするframeを指定

ax.set_xlabel('sec')
ax.set_ylabel('frequency [Hz]')
ax.imshow(
	np.flipud(np.array(spectrogram).T),
	extent=[0, duration, 0, 8000/8],
	aspect='auto',
	interpolation='nearest'
)
ax.plot(np.linspace(0,(len(x)-size_frame)/16000, len(fundamental_frequency)), fundamental_frequency, color='red')
aiueo_list = np.array(aiueo_list) * 100 + 100
ax.plot(np.linspace(0, len(x)/SR-0.8, len(aiueo_list[80:])), aiueo_list[80:], color="white")
print( len(aiueo_list))
# 続いて右側のy軸を追加して，音量を重ねて描画
axtw = ax.twinx()
axtw.set_ylabel('volume [dB]')
x_data = np.linspace(0, duration, len(volume))
axtw.plot(x_data, volume, c='y')

canvas.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

# スライドバーの値が変更されたときに呼び出されるコールバック関数
# ここで右側のグラフに
# vはスライドバーの値
def _draw_spectrum(v):

	# スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
	index = int((len(spectrogram_all)-1) * (float(v) / duration))
	spectrum = spectrogram_all[index]

	# 直前のスペクトル描画を削除し，新たなスペクトルを描画
	plt.cla()
	x_data = np.fft.rfftfreq(size_frame, d=1/SR)
	ax2.plot(x_data, spectrum)
	ax2.set_ylim(-10, 5)
	ax2.set_xlim(0, SR/2)
	ax2.set_ylabel('amblitude')
	ax2.set_xlabel('frequency [Hz]')
	canvas2.draw()

# スペクトルを表示する領域を確保
# ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
fig2, ax2 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
canvas2.get_tk_widget().pack(side="top")	# "top"は上部方向にウィジェットを積むことを意味する

# スライドバーを作成
scale = tkinter.Scale(
	command=_draw_spectrum,		# ここにコールバック関数を指定
	master=frame2,				# 表示するフレーム
	from_=0,					# 最小値
	to=duration,				# 最大値
	resolution=size_shift/SR,	# 刻み幅
	label=u'スペクトルを表示する時間[sec]',
	orient=tkinter.HORIZONTAL,	# 横方向にスライド
	length=600,					# 横サイズ
	width=50,					# 縦サイズ
	font=("", 20)				# フォントサイズは20pxに設定
)
scale.pack(side="top")

# TkinterのGUI表示を開始
tkinter.mainloop()

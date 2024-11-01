import librosa
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import NMF
import sounddevice as sd
import tkinter
# MatplotlibをTkinterで使用するために必要
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math
from tkinter import ttk, filedialog
import pygame
import time
import threading

# 周波数からノートナンバーへ変換（notenumber.pyより）
def hz2nn(frequency):
	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

def chroma_vector(spectrum, frequencies):
	cv = np.zeros(12)
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn % 12] += abs(s)
	return cv

size_frame = 4096	# フレームサイズ
SR = 16000			# サンプリングレート
size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)

# 音声ファイルを読み込む
x, _ = librosa.load('rec/nmf_piano_sample.wav', sr=SR)

duration = len(x) / SR                      # ファイルサイズ（秒）
hamming_window = np.hamming(size_frame)     # ハミング窓
spectrogram = []                            # スペクトログラムを保存するlist
spectrogram1 = [] 
chromagram = []                             # クロマグラフを保存するlist
chord = []                                  # 和音を保存するリスト

a_root = 1
a_3 = 0.5
a_5 = 0.8

# フレーム毎にスペクトルを計算
for i in np.arange(0, len(x)-size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    x_frame = x[idx : idx+size_frame]
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec) + 1)
    spectrogram1.append(fft_log_abs_spec)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)

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

Y = np.array(spectrogram1).T # f * t
f,t = np.shape(Y)
k = 3
# H, U を作成 Y = H * U
np.random.seed(91)
H = np.abs(np.random.uniform(low=0, high=1, size=(f,k)))
U = np.abs(np.random.uniform(low=0, high=1, size=(k,t)))

# 更新式
for i in range(100):
    H = H * (Y @ U.T) / (U @ (H @ U).T).T
    U = U * (Y.T @ H).T / (H.T @ (H @ U))
    NMF = H @ U
    D = np.mean(np.square(Y-NMF))
    print(D)
    if np.abs(D) < 0.01:
        break

spectrograms = []
y = []

# 位相復元
for i in range(k):
    HH = np.zeros_like(H)
    HH.T[i] = H.T[i]
    spectrograms.append(HH @ U)
    y.append(librosa.istft(spectrograms[i]))

# Pygame初期化
pygame.mixer.init()
playing = False # 再生状態の変数
# Tkinterを初期化
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-ASSIGNMENT1-1")

# Tkinterのウィジェットを階層的に管理するためにFrameを使用
# frame1 ... スペクトログラムを表示
# frame2 ... Scale（スライドバー）とスペクトルを表示
frame0 = tkinter.Frame(root)
frame1 = tkinter.Frame(frame0)
frame2 = tkinter.Frame(root)
frame3 = tkinter.Frame(root)
frame4 = tkinter.Frame(root)
frame5 = tkinter.Frame(root)
frame0.pack(side="top")
frame1.pack(side="left")
frame2.pack(side="left")
frame3.pack(side="left")
frame4.pack(side="left")
frame5.pack(side="left")

# まずはスペクトログラムを描画
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame0)	# masterに対象とするframeを指定
plt.title("spectrogram")
plt.xlabel('sec')
plt.ylabel('frequency [Hz]')
plt.imshow(
	np.flipud(np.array(spectrogram).T),
	extent=[0, duration, 0, 8000],
	aspect='auto',
	interpolation='nearest'
)
canvas.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

# クロマグラムを描画
fig1, ax1 = plt.subplots()
canvas1 = FigureCanvasTkAgg(fig1, master=frame1)	# masterに対象とするframeを指定
plt.title("chromagram")
plt.xlabel('time(s)')				# x軸のラベルを設定
plt.ylabel('chroma')				# y軸のラベルを設定
plt.imshow(
	np.flipud(np.array(chromagram).T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, len(x)/16000, 0, 24],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.plot(np.linspace(0, len(x)/16000,len(chord)),chord, color="white")
plt.yticks(range(0,24),['C','Cm','C#','C#m','D','Dm','D#','D#m','E','Em','F','Fm','F#','F#m','G','Gm','G#','G#m','A','Am','A#','A#m','B','Bm'])
canvas1.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

fig2, ax2 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)	# masterに対象とするframeを指定
plt.title("H")
plt.xlabel('frequency(Hz)')					# x軸のラベルを設定
plt.ylabel('num')		# y軸のラベルを設定
plt.imshow(
	np.flipud(H.T),		# 画像とみなすために，データを転置して上下反転
	extent=[0, SR/2, 0,k],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
canvas2.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

fig3, ax3 = plt.subplots()
canvas3 = FigureCanvasTkAgg(fig3, master=frame3)	# masterに対象とするframeを指定
plt.title("U")
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('num')		# y軸のラベルを設定
plt.imshow(
	np.flipud(U),		# 画像とみなすために，データを転置して上下反転
	extent=[0, len(x)/SR, 0,k],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
canvas3.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

# スライドバーの値が変更されたときに呼び出されるコールバック関数
# ここで右側のグラフに
# vはスライドバーの値
def _draw_spectrum(v):

    # スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
    index = int((len(spectrogram)-1) * (float(v) / duration))
    spectrum = spectrogram[index]

    # 直前のスペクトル描画を削除し，新たなスペクトルを描画
    # plt.cla()
    axA.cla()
    x_data = np.fft.rfftfreq(size_frame, d=1/SR)
    axA.plot(x_data, spectrum)
    axA.set_ylim(-10, 5)
    axA.set_xlim(0, SR/2)
    axA.set_ylabel('amblitude')
    axA.set_xlabel('frequency [Hz]')
    canvasA.draw()

# スペクトルを表示する領域を確保
# ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
figA, axA = plt.subplots()
canvasA = FigureCanvasTkAgg(figA, master=frame4)
canvasA.get_tk_widget().pack(side="top")	# "top"は上部方向にウィジェットを積むことを意味する

# スライドバーを作成
scale = tkinter.Scale(
	command=_draw_spectrum,		# ここにコールバック関数を指定
	master=frame4,				# 表示するフレーム
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

fig5, ax5 = plt.subplots()
canvas5 = FigureCanvasTkAgg(fig5, master=frame5)
# 波形データをプロット
ax5.clear()
ax5.plot(np.linspace(0, duration, len(x)), x, color='blue')
ax5.set_xlim(0, duration)
ax5.set_ylim(np.min(x), np.max(x))
ax5.set_xlabel("Time [s]")
ax5.set_ylabel("Amplitude")
canvas.draw()
canvas5.get_tk_widget().pack(side="top")	# "top"は上部方向にウィジェットを積むことを意味する

# 再生ボタン
def play_audio():
    sd.play(x,SR)
play_button = tkinter.Button(frame1, text="Play Audio", command=play_audio, width=10, height=5,font=("MSゴシック", "20", "bold"))
play_button.pack(pady=30)

# # 再生ボタン
# def play_audio1():
#     sd.play(y[0],SR)
# play_button = tkinter.Button(root, text="Play Audio1", command=play_audio1)
# play_button.pack(pady=10)

# # 再生ボタン
# def play_audio2():
#     sd.play(y[1],SR)
# play_button = tkinter.Button(root, text="Play Audio2", command=play_audio2)
# play_button.pack(pady=10)

# TkinterのGUI表示を開始
tkinter.mainloop()

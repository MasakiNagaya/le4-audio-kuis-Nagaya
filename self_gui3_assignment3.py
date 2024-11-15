#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 簡易カラオケシステム
#
# mp3ファイルを別スレッドで再生しつつ
# マイクからの音声入力に対してスペクトログラムとパワーを計算して表示する
# 上記をリアルタイムで逐次処理を行う
#

# ライブラリの読み込み
import pyaudio
import numpy as np
import threading
import time

# matplotlib関連
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI関連
import tkinter
from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)
import math

# mp3ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks

SAMPLING_RATE = 16000	# サンプリングレート
FRAME_SIZE = 2048		# フレームサイズ
SHIFT_SIZE = int(SAMPLING_RATE / 20)	# サイズシフト # 今回は0.05秒
SPECTRUM_MIN = -5	# スペクトルをカラー表示する際に色の範囲を正規化するためにスペクトルの最小値と最大値を指定
SPECTRUM_MAX = 1	# スペクトルの値がこの範囲を超えると，同じ色になってしまう
VOLUME_MIN = -120	# 音量を表示する際の値の範囲
VOLUME_MAX = -10
EPSILON = 1e-10	# log10を計算する際に，引数が0にならないようにするためにこの値を足す
hamming_window = np.hamming(FRAME_SIZE)	# ハミング窓
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)	# グラフに表示する縦軸方向のデータ数
NUM_DATA_SHOWN = 100		# グラフに表示する横軸方向のデータ数
is_gui_running = False		# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
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
# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

#
# (1) GUI / グラフ描画の処理
#

# ここでは matplotlib animation を用いて描画する
# 毎回 figure や ax を初期化すると処理時間がかかるため
# データを更新したら，それに従って必要な部分のみ再描画することでリアルタイム処理を実現する

# matplotlib animation によって呼び出される関数
# ここでは最新のスペクトログラムと音量のデータを格納する
# 再描画はmatplotlib animationが行う
def animate(frame_index):

	ax1_sub.set_array(spectrogram_data)

	# この上の処理を下記のようにすれば楽曲のスペクトログラムが表示される
	# ax1_sub.set_array(spectrogram_data_music)
	ax1_sub_f.set_data(time_x_data, f_frequency_data)
	ax1_sub_f_music.set_data(time_x_data, f_frequency_data_music)

	ax2_sub.set_data(time_x_data, volume_data)

	volume_meter_canvas.delete("all")
	volume_meter_canvas.create_rectangle(0, 400 * (1-v_level), 20 , 400, fill=lm_color)
	
	return ax1_sub, ax1_sub_f, ax1_sub_f_music, ax2_sub

# GUIで表示するための処理（Tkinter）
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-NAGAYA")

main_frame = tkinter.Frame(root)
main_frame.pack(side="left")

title_music = tkinter.Label(main_frame, text="曲名", font=("", 25))
title_music.pack(pady=20)  # 上下に余白を設定して配置
lyrics_music = tkinter.Label(main_frame, text="歌詞", font=("", 25))
lyrics_music.pack(pady=20)  # 上下に余白を設定して配置

# スペクトログラムを描画
fig, ax1 = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=main_frame)

# 横軸の値のデータ
time_x_data = np.linspace(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE), NUM_DATA_SHOWN)
# 縦軸の値のデータ
freq_y_data = np.linspace(8000/MAX_NUM_SPECTROGRAM, 8000, MAX_NUM_SPECTROGRAM)

# とりあえず初期値（ゼロ）のスペクトログラムと音量のデータを作成
# この numpy array にデータが更新されていく
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
volume_data = np.zeros(len(time_x_data))
f_frequency_data = np.zeros(len(time_x_data))

# 楽曲のスペクトログラムを格納するデータ（このサンプルでは計算のみ）
spectrogram_data_music = np.zeros((len(freq_y_data), len(time_x_data)))
f_frequency_data_music = np.zeros(len(time_x_data))

# スペクトログラムを描画する際に横軸と縦軸のデータを行列にしておく必要がある
# これは下記の matplotlib の pcolormesh の仕様のため
X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
	for idx_t, t_v in enumerate(time_x_data):
		X[idx_f, idx_t] = t_v
		Y[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub = ax1.pcolormesh(
	X,
	Y,
	spectrogram_data,
	shading='nearest',	# 描画スタイル
	cmap='jet',			# カラーマップ
	norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)	# 値の最小値と最大値を指定して，それに色を合わせる
)

ax1_sub_f, = ax1.plot(time_x_data, f_frequency_data, c='white')
ax1_sub_f_music, = ax1.plot(time_x_data, f_frequency_data_music, c='red', lw=5)

# 音量を表示するために反転した軸を作成
ax2 = ax1.twinx()

# 音量をプロットする
# 戻り値はデータの更新 & 再描画のために必要
ax2_sub, = ax2.plot(time_x_data, volume_data, c='y')
# ax2_sub, = ax2.plot(time_x_data, f_frequency_data, c='y')

# ラベルの設定
ax1.set_xlabel('sec')				# x軸のラベルを設定
ax1.set_ylabel('frequency [Hz]')	# y軸のラベルを設定
ax2.set_ylabel('volume [dB]')		# 反対側のy軸のラベルを設定

# 音量を表示する際の値の範囲を設定
ax2.set_ylim([VOLUME_MIN, VOLUME_MAX])
# ax2.set_ylim([0, 500])

v_level = 10
lm_color = "green"
volume_frame = tkinter.Frame(root)
volume_frame.pack(side="left")
volume_meter_canvas = tkinter.Canvas(volume_frame, width=20, height=400, bg="black")
volume_meter_canvas.pack(pady=5)

# maplotlib animationを設定
ani = animation.FuncAnimation(
	fig,
	animate,		# 再描画のために呼び出される関数
	interval=100,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
	blit=True		# blitting処理を行うため描画処理が速くなる（環境によってうまく動作しない場合はこれをFalseにしてみること）
)

# matplotlib を GUI(Tkinter) に追加する
toolbar = NavigationToolbar2Tk(canvas, main_frame)
canvas.get_tk_widget().pack()

# 再生位置をテキストで表示するためのラベルを作成
text = tkinter.StringVar()
text.set('0.0')
label = tkinter.Label(master=main_frame, textvariable=text, font=("", 30))
label.pack()

score_label = tkinter.Label(master=main_frame, text="score", font=("", 30))
score_label.pack()

# 終了ボタンが押されたときに呼び出される関数
# ここではGUIを終了する
def _quit():
	root.quit()
	root.destroy()

# 終了ボタンを作成
bottun_frame = tkinter.Frame(main_frame)
bottun_frame.pack(side="bottom")
button = tkinter.Button(master=bottun_frame, text="アプリの終了", command=_quit, font=("", 30))
button.pack(side="right")

#
# (2) マイク入力のための処理
#

x_stacked_data = np.array([])
# フレーム毎に呼び出される関数
def input_callback(in_data, frame_count, time_info, status_flags):
	
	# この関数は別スレッドで実行するため
	# メインスレッドで定義した以下の２つの numpy array を利用できるように global 宣言する
	# これらにはフレーム毎のスペクトルと音量のデータが格納される
	global x_stacked_data, spectrogram_data, volume_data, f_frequency_data, v_level, lm_color

	# 現在のフレームの音声データをnumpy arrayに変換
	x_current_frame = np.frombuffer(in_data, dtype=np.float32)

	# 現在のフレームとこれまでに入力されたフレームを連結
	x_stacked_data = np.concatenate([x_stacked_data, x_current_frame])

	# フレームサイズ分のデータがあれば処理を行う
	if len(x_stacked_data) >= FRAME_SIZE:
		
		# フレームサイズからはみ出した過去のデータは捨てる
		x_stacked_data = x_stacked_data[len(x_stacked_data)-FRAME_SIZE:]

		# スペクトルを計算
		fft_spec = np.fft.rfft(x_stacked_data * hamming_window)
		fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

		# ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
		# 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
		spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
		spectrogram_data[:, -1] = fft_log_abs_spec

		# 音量も同様の処理
		vol = 20 * np.log10(np.mean(x_current_frame ** 2) + EPSILON)
		volume_data = np.roll(volume_data, -1)
		volume_data[-1] = vol
		v_level = (vol - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN)
		if v_level < 0.3:
			lm_color = "green"
		elif v_level < 0.7:
			lm_color = "yellow"
		else:
			lm_color = "red"

		# 音声の基本周波数
		f_frequency = 0
		autocorr = np.correlate(x_stacked_data, x_stacked_data, 'full')
		autocorr = autocorr [len (autocorr ) // 2 : ]
		peakindices = [i for i in range (len (autocorr)-1) if is_peak (autocorr, i)]
		peakindices = [i for i in peakindices if i != 0]
		if peakindices!=[]:
			max_peak_index = max(peakindices , key=lambda index: autocorr [index])
			tau = max_peak_index/SAMPLING_RATE
			# zero_cross
			zero_cross_rate = zero_cross(x_stacked_data) / (FRAME_SIZE/SAMPLING_RATE)
			if zero_cross_rate > (1/tau)*0.5 and zero_cross_rate < (1/tau)*11:
				if (1/tau)<1000:
					f_frequency = 1/tau
			f_frequency = 1/tau #音声だけなら消す
		f_frequency_data = np.roll(f_frequency_data, -1)
		# print(f_frequency, zero_cross_rate)
		# print(v_level)
		f_frequency_data[-1] = f_frequency
	
	# 戻り値は pyaudio の仕様に従うこと
	return None, pyaudio.paContinue

# マイクからの音声入力にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
# 【注意】シフトサイズごとに指定された関数が呼び出される
p = pyaudio.PyAudio()
stream = p.open(
	format = pyaudio.paFloat32,
	channels = 1,
	rate = SAMPLING_RATE,
	input = True,						# ここをTrueにするとマイクからの入力になる 
	frames_per_buffer = SHIFT_SIZE,		# シフトサイズ
	stream_callback = input_callback	# ここでした関数がマイク入力の度に呼び出される（frame_per_bufferで指定した単位で）
)


#
# (3) mp3ファイル音楽を再生する処理
#

# mp3ファイル名
filename = 'rec/mp3/nhk_hotaru.mp3'
# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_mp3(filename)
# txtファイル名
note_filename = 'data/hotaru_note.txt'
note_data = np.loadtxt(note_filename)
fhz_data = nn2hz(note_data - 7)
# txtファイル名
txt_filename = 'data/hotaru_text.txt'
txt_data = np.genfromtxt(txt_filename, dtype=str, delimiter='\n', encoding='utf-8')
# title
title_music.config(text="蛍の光")
# scale
scale = 37.5
lyric_scale = 8 * 4

def select_song():
	global filename, audio_data, fhz_data, txt_data, scale, lyric_scale
	if selected_song.get() == song_list[0]:
		# mp3ファイル名
		filename = 'rec/mp3/nhk_hotaru.mp3'
		# pydubを使用して音楽ファイルを読み込む
		audio_data = AudioSegment.from_mp3(filename)
		# txtファイル名
		note_filename = 'data/hotaru_note.txt'
		note_data = np.loadtxt(note_filename)
		fhz_data = nn2hz(note_data)
		# txtファイル名
		txt_filename = 'data/hotaru_text.txt'
		txt_data = np.genfromtxt(txt_filename, dtype=str, delimiter='\n', encoding='utf-8')
		# title
		title_music.config(text="蛍の光")
		# scale
		scale = 37.7
		lyric_scale = 8 * 4
	elif selected_song.get() == song_list[1]:
		# mp3ファイル名
		filename = 'rec/mp3/nhk_akatommbo.mp3'
		# pydubを使用して音楽ファイルを読み込む
		audio_data = AudioSegment.from_mp3(filename)
		# txtファイル名
		note_filename = 'data/akatommbo_note.txt'
		note_data = np.loadtxt(note_filename)
		fhz_data = nn2hz(note_data)
		# txtファイル名
		txt_filename = 'data/akatommbo_text.txt'
		txt_data = np.genfromtxt(txt_filename, dtype=str, delimiter='\n', encoding='utf-8')
		# title
		title_music.config(text="赤とんぼ")
		# scale
		scale = 41
		lyric_scale = 6
	elif selected_song.get() == song_list[2]:
		# mp3ファイル名
		filename = 'rec/mp3/furusato.mp3'
		# pydubを使用して音楽ファイルを読み込む
		audio_data = AudioSegment.from_mp3(filename)
		# txtファイル名
		note_filename = 'data/furusato_note.txt'
		note_data = np.loadtxt(note_filename)
		fhz_data = nn2hz(note_data)
		# txtファイル名
		txt_filename = 'data/furusato_text.txt'
		txt_data = np.genfromtxt(txt_filename, dtype=str, delimiter='\n', encoding='utf-8')
		# title
		title_music.config(text="ふるさと")
		# scale
		scale = 31.25
		lyric_scale = 6 * 4

# ラジオボタンで曲を選択
song_list = ["蛍の光", "赤とんぼ", "ふるさと"]
selected_song = tkinter.StringVar(value=song_list[0])
radio_frame = tkinter.Frame(root)
radio_frame.pack(pady=10, side="right")
for song in song_list:
	radio = tkinter.Radiobutton(
		radio_frame, text=song, variable=selected_song,
		value=song, font=("", 25)
	)
	radio.pack(anchor="w")
button_song = tkinter.Button(master=radio_frame, text="曲の決定", command=select_song, font=("", 30))
button_song.pack()

cheet_flag = False
def cheet():
	global cheet_flag
	cheet_flag = not cheet_flag
	if cheet_flag:
		button_cheet.config(text="接待モード")
	else:
		button_cheet.config(text="通常モード")

button_cheet = tkinter.Button(master=radio_frame, text="通常モード", command=cheet, font=("", 30))
button_cheet.pack()

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
	format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
	channels = audio_data.channels,								# チャネル数
	rate = audio_data.frame_rate,								# サンプリングレート
	output = True												# 出力モードに設定
)

# 楽曲のデータを格納
x_stacked_data_music = np.array([])

# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため
def play_music():

	# この関数は別スレッドで実行するため
	# メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
	global is_gui_running, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music, f_frequency_data_music

	# pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
	# 第二引数には何ミリ秒毎に読み込むかを指定
	# ここでは10ミリ秒ごとに読み込む

	size_frame_music = 10	# 10ミリ秒毎に読み込む

	idx = 0

	# make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
	#
	# なぜ再生するだけのためにフレーム毎の処理をするのか？
	# 音楽ファイルに対しても何らかの処理を行えるようにするため（このサンプルプログラムでは行っていない）
	# おまけに，再生位置も計算することができる
	for chunk in make_chunks(audio_data, size_frame_music):
		
		# GUIが終了してれば，この関数の処理も終了する
		if is_gui_running == False:
			break

		if is_paused:
			time.sleep(0.01)
		else:

			# pyaudioの再生ストリームに切り出した音楽データを流し込む
			# 再生が完了するまで処理はここでブロックされる
			stream_play.write(chunk._data)
			
			# 現在の再生位置を計算（単位は秒）
			now_playing_sec = (idx * size_frame_music) / 1000.
			
			idx += 1

			#
			# 【補足】
			# 楽曲のスペクトログラムを計算する場合には下記のように楽曲のデータを受け取る
			# ただし，音声データの値は -1.0~1.0 ではなく，16bit の整数値であるので正規化を施している
			# また十分なサイズの音声データを確保してからfftを実行すること
			# 楽曲が44.1kHzの場合，44100 / (1000/10) のサイズのデータとなる
			# 以下では処理のみを行い，表示はしない．表示をするには animate 関数の中身を変更すること 
			
			# データの取得
			data_music = np.array(chunk.get_array_of_samples())
			
			# 正規化
			data_music = data_music / np.iinfo(np.int32).max	

			#
			# 以下はマイク入力のときと同様
			#

			# 現在のフレームとこれまでに入力されたフレームを連結
			x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])

			# フレームサイズ分のデータがあれば処理を行う
			if len(x_stacked_data_music) >= FRAME_SIZE:
				
				# フレームサイズからはみ出した過去のデータは捨てる
				x_stacked_data_music = x_stacked_data_music[len(x_stacked_data_music)-FRAME_SIZE:]

				# スペクトルを計算
				fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window)
				fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

				# ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
				# 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
				spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis=1)
				spectrogram_data_music[:, -1] = fft_log_abs_spec

				# 音声の基本周波数
				index = int(min(idx/scale,len(fhz_data)-1))
				f_frequency_data_music = np.roll(f_frequency_data_music, -1)
				f_frequency_data_music[-1] = fhz_data[index]

				# lyrics
				index = int(min(idx/(scale*lyric_scale) ,len(txt_data)-1))
				lyrics_music.config(text=txt_data[index])
	
score = 0
full_score = 0
# 再生時間の表示を随時更新する関数
def update_gui_text():
	global is_gui_running, now_playing_sec, text, is_paused, score, full_score
	score = 0
	full_score = 0
	while True:
		if not is_paused:
			# GUIが表示されていれば再生位置（秒）をテキストとしてGUI上に表示
			if is_gui_running:
				text.set('%.3f' % now_playing_sec)
				# score
				if f_frequency_data_music[-1] != 0 and f_frequency_data[-1] != 0:
					full_score += 1
					if np.abs(f_frequency_data_music[-1] - f_frequency_data[-1]) < 10 or np.abs(f_frequency_data_music[-1]/2 - f_frequency_data[-1]) < 10 or np.abs(f_frequency_data_music[-1]/4 - f_frequency_data[-1]) < 10:
						if cheet_flag:
							score += 10
						else:
							score += 1
						score_label.config(text="score: " + str(score) + "/" + str(full_score) + " => " + str(int(100 * score/full_score)) + "点")
		# 0.01秒ごとに更新
		time.sleep(0.01)

# 再生時間を表す
now_playing_sec = 0.0

# 音楽を再生するパートを関数化したので，それを別スレッドで（GUIのため）再生開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため

#
# (4) 全体の処理を実行
#
START = True
STOP = False
ss_flag = False
is_paused = False
def _start():
	global is_gui_running, t_play_music, t_update_gui, p_play, stream_play, ss_flag, score, full_score

	if ss_flag == START:
		return
	
	ss_flag = START
	score = 0
	full_score = 0

	# GUIの開始フラグをTrueに
	is_gui_running = True

	# 音声ファイルの再生にはpyaudioを使用
	# ここではpyaudioの再生ストリームを作成
	p_play = pyaudio.PyAudio()
	stream_play = p_play.open(
		format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
		channels = audio_data.channels,								# チャネル数
		rate = audio_data.frame_rate,								# サンプリングレート
		output = True												# 出力モードに設定
	)

	# 新しいスレッドを作成して開始
	t_play_music = threading.Thread(target=play_music)
	t_play_music.setDaemon(True)
	t_update_gui = threading.Thread(target=update_gui_text)
	t_update_gui.setDaemon(True)

	# 上記で設定したスレッドを開始（直前のフラグを立ててから）
	t_play_music.start()
	t_update_gui.start()

def _pause():
	global is_gui_running, is_paused
	# 一時停止/再開を切り替え
	if is_gui_running:
		is_paused = not is_paused
		if is_paused:
			button_pause.config(text="再開")
		else:
			button_pause.config(text="停止")

def _stop():
	global is_gui_running, ss_flag, stream_play, p_play

	if ss_flag == STOP:
		return

	ss_flag = STOP

	# GUIの開始フラグをFalseに = 音楽再生スレッドのループを終了
	is_gui_running = False

	# 終了処理
	stream_play.stop_stream()
	stream_play.close()
	p_play.terminate()

button_start = tkinter.Button(master=bottun_frame, text="音楽開始", command=_start, font=("", 30))
button_start.pack(side="left")
button_pause = tkinter.Button(master=bottun_frame, text="ポーズ", command=_pause, font=("", 30))
button_pause.pack(side="left")
button_stop = tkinter.Button(master=bottun_frame, text="音楽停止", command=_stop, font=("", 30))
button_stop.pack(side="left")

# GUIを開始，GUIが表示されている間は処理はここでストップ
tkinter.mainloop()

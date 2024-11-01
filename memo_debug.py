import numpy as np

# x= np.ones(100)
# num = int(np.log2(len(x)))
# print(np.exp2(num+1))
# print(x)

# while(len(x)<np.exp2(num+1)):
#     x = np.append(x,0)

# index = np.arange(len(x))

# print(len(x))
# print(num)
# print(index)

# x= np.arange(6).reshape(2,3)
# print(np.size(x))
# print(len(x))

# x=[1,2,3]
# x=x*10 +3
# print(x)

# a = np.array([[150, 160, 170, 180, 190], 
#               [55, 60, 65, 60, 70]])
# print(np.cov(a))

# spectrogram = np.abs(np.random.uniform(low=0, high=1, size=(5,6)))

# Y = np.array(spectrogram.T) # f * t
# f,t = np.shape(Y)
# k = 4
# # H, U を作成 Y = H * U
# np.random.seed(91)
# H = np.abs(np.random.uniform(low=0, high=1, size=(f,k)))
# U = np.abs(np.random.uniform(low=0, high=1, size=(k,t)))

# print(H.T[0])

# NMF = H @ U
# print(np.mean(Y-NMF))

# # 更新式
# for i in range(100):
#     H = H * (Y @ U.T) / (U @ (H @ U).T).T
#     U = U * (Y.T @ H).T / (H.T @ (H @ U))

# NMF = H @ U
# print(np.mean(Y-NMF))

# for i in range(k):
#     HH = np.zeros_like(H)
#     HH.T[i] = H.T[i]
#     print(HH)
#     print(HH @ U)

# x = 1
# def f (x):
#     x=x+1
#     print(x)
# f(x)
# print(x)

# a = np.arange(6).reshape(2, 3)

# np.savetxt('data/test1.txt', a)

# b = np.loadtxt('data/test1.txt')
# print(b)


# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import filedialog
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from sklearn.decomposition import NMF
# import sounddevice as sd
# import math

# # 音声ファイルのロード
# def load_file():
#     SR = 16000
#     filepath = filedialog.askopenfilename()
#     if filepath:
#         x, sr = librosa.load(filepath, sr=None)
#         return x, sr
#     return None, None

# # NMFを用いた音声と音楽の分離
# def separate_audio(S):
#     model = NMF(n_components=2, init='random', random_state=0)
#     W = model.fit_transform(np.abs(S))  # 基底行列
#     H = model.components_  # 係数行列
#     return W, H

# # NMFで分離した成分を再構成
# def reconstruct_audio(W, H, component):
#     S_reconstructed = np.dot(W[:, component].reshape(-1, 1), H[component, :].reshape(1, -1))
#     return S_reconstructed

# # 音声を再生
# def play_audio(y, sr):
#     sd.play(y, sr)

# # 周波数からノートナンバーへ変換（notenumber.pyより）
# def hz2nn(frequency):
# 	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

# def chroma_vector(spectrum, frequencies):
# 	cv = np.zeros(12)
# 	for s, f in zip (spectrum , frequencies):
# 		nn = hz2nn(f)
# 		cv[nn % 12] += abs(s)
# 	return cv

# # GUI設定
# class AudioApp:
#     def __init__(self, root):
#         self.root = root
        
#         # 音声ファイルのロードボタン
#         self.load_button = tk.Button(root, text="Load Audio File", command=self.load_audio)
#         self.load_button.pack()

#         # 再生ボタン
#         self.play_music_button = tk.Button(root, text="Play Music", command=self.play_music)
#         self.play_music_button.pack()
#         self.play_voice_button = tk.Button(root, text="Play Voice", command=self.play_voice)
#         self.play_voice_button.pack()

#         # グラフ描画領域
#         self.fig, self.ax = plt.subplots(121)
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
#         self.canvas.get_tk_widget().pack()
        
#         # self.fig2, self.ax2 = plt.subplots(122)
#         # self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.root)
#         # self.canvas2.get_tk_widget().pack()

#     # 音声ファイルのロード
#     def load_audio(self):
#         self.audio_data, self.sr = load_file()
#         if self.audio_data is not None:
#             self.display_spectrogram()

#     # スペクトログラムの表示
#     def display_spectrogram(self):
#         S = np.abs(librosa.stft(self.audio_data))
#         librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=self.sr, x_axis='time', y_axis='log', ax=self.ax)
#         self.ax.set_title("Spectrogram")
#         self.canvas.draw()

#         # 音声と音楽の分離
#         W, H = separate_audio(S)
#         self.music_component = reconstruct_audio(W, H, component=0)
#         self.voice_component = reconstruct_audio(W, H, component=1)

#     def load_and_display_initial(self):
#         size_frame = 4096	        # フレームサイズ
#         SR = 16000			        # サンプリングレート
#         size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)
#         # 音声ファイルを読み込む
#         x = self.audio_data
#         duration = len(x) / SR                      # ファイルサイズ（秒）
#         hamming_window = np.hamming(size_frame)     # ハミング窓
#         spectrogram = []                            # スペクトログラムを保存するlist
#         chromagram = []                             # クロマグラフを保存するlist
#         chord = []                                  # 和音を保存するリスト

#         a_root = 1
#         a_3 = 0.5
#         a_5 = 0.8

#         # フレーム毎にスペクトルを計算
#         for i in np.arange(0, len(x)-size_frame, size_shift): 
#             # 該当フレームのデータを取得
#             idx = int(i)	# arangeのインデクスはfloatなのでintに変換
#             x_frame = x[idx : idx+size_frame]
#             # スペクトル
#             fft_spec = np.fft.rfft(x_frame * hamming_window)
#             fft_log_abs_spec = np.log(np.abs(fft_spec))
#             spectrogram.append(fft_log_abs_spec)
#             # クロマグラフ
#             frequencies = np.linspace((SR/2)/len(fft_spec), SR/2, len(fft_spec))
#             chroma = chroma_vector(fft_spec,frequencies)
#             chromagram.append(chroma)
#             #和音
#             chroma = np.append(chroma,chroma)
#             l=[]
#             for i in range(0,12):
#                 l.append(a_root*chroma[i] + a_3*chroma[i+4] + a_5*chroma[i+7])
#                 l.append(a_root*chroma[i] + a_3*chroma[i+3] + a_5*chroma[i+7])
#             chord.append(np.argmax(np.array(l)))
        
#         self.ax.set_title("Spectrogram")
#         self.ax.set_xlabel('sec')
#         self.ax.set_ylabel('frequency [Hz]')
#         self.ax.imshow(
#         np.flipud(np.array(spectrogram).T),
#             extent=[0, duration, 0, 8000],
#             aspect='auto',
#             interpolation='nearest'
#         )
#         self.canvas.draw()
#         self.ax2.set_title("Spectrogram")
#         self.ax2.set_xlabel('sec')
#         self.ax2.set_ylabel('frequency [Hz]')
#         self.ax2.imshow(
#         np.flipud(np.array(chromagram).T),
#             extent=[0, duration, 0, 24],
#             aspect='auto',
#             interpolation='nearest'
#         )
#         self.ax.plot(np.linspace(0, len(x)/16000,len(chord)),chord, color="white")
#         self.set_yticklabels(range(0,24), ['C','Cm','C#','C#m','D','Dm','D#','D#m','E','Em','F','Fm','F#','F#m','G','Gm','G#','G#m','A','Am','A#','A#m','B','Bm'])
#         self.canvas2.draw()

#     # 音楽部分の再生
#     def play_music(self):
#         if self.music_component is not None:
#             music_signal = librosa.istft(self.music_component)
#             play_audio(music_signal, self.sr)

#     # 音声部分の再生
#     def play_voice(self):
#         if self.voice_component is not None:
#             voice_signal = librosa.istft(self.voice_component)
#             play_audio(voice_signal, self.sr)

# # Tkinterアプリケーションの起動
# root = tk.Tk()
# root.wm_title("EXP4-AUDIO-ASSIGNMENT1")
# app = AudioApp(root)
# root.mainloop()

# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# from tkinter import filedialog
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from sklearn.decomposition import NMF
# import sounddevice as sd
# import tkinter
# # MatplotlibをTkinterで使用するために必要
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# import math
# from tkinter import ttk, filedialog
# import tkinter as tk 
# import pygame
# import time
# import threading

# root = tk.Tk()
# root.title("ASSIGNMENT2 - NAGAYA Masaki")
# root.geometry("400x200")#サイズ

# #text
# label = tk.Label(root, text="こんにちは")
# label.pack()

# #textbox
# txtBox = tk.Entry()
# txtBox.configure(state='normal', width=50)
# txtBox.pack()


# #bottun
# def outputWords(event):
# 	txtBox.insert(tk.END, 'Hello!!!')

# txtBox = tk.Entry()
# txtBox.configure(state='normal', width=50)
# txtBox.pack()

# button = tk.Button(text='ボタン', width=30)
# button.place(x=90, y=120)
# button.bind('<Button-1>', outputWords)


# root.mainloop()

import tkinter as tk
from tkinter import ttk, filedialog
import pygame
import time
import threading

class MusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Music Player with Seek Bar")
        
        # Pygame初期化
        pygame.mixer.init()

        # UIセットアップ
        self.setup_ui()

        # 再生状態の変数
        self.playing = False

    def setup_ui(self):
        # ファイルを開くボタン
        self.load_button = tk.Button(self.root, text="Load Music", command=self.load_music)
        self.load_button.pack(pady=10)

        # 再生ボタン
        self.play_button = tk.Button(self.root, text="Play", command=self.play_music)
        self.play_button.pack(pady=10)

        # 停止ボタン
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_music)
        self.stop_button.pack(pady=10)

        # シークバー
        self.seek_bar = ttk.Scale(self.root, from_=0, to=100, orient="horizontal", command=self.seek_music)
        self.seek_bar.pack(pady=10, fill='x', padx=20)
        
        # 現在の再生時間ラベル
        self.current_time_label = tk.Label(self.root, text="0:00")
        self.current_time_label.pack(side="left", padx=20)

        # 総再生時間ラベル
        self.total_time_label = tk.Label(self.root, text="0:00")
        self.total_time_label.pack(side="right", padx=20)

    def load_music(self):
        # 音楽ファイルの選択
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if self.file_path:
            pygame.mixer.music.load(self.file_path)
            pygame.mixer.music.set_endevent(pygame.USEREVENT)
            
            # 音楽の長さを取得
            self.total_duration = pygame.mixer.Sound(self.file_path).get_length()
            self.total_time_label.config(text=self.format_time(self.total_duration))
            self.seek_bar.config(to=self.total_duration)

    def play_music(self):
        if self.file_path and not self.playing:
            pygame.mixer.music.play()
            self.playing = True
            self.update_seek_bar()

    def stop_music(self):
        # 再生を停止して再生位置をリセット
        pygame.mixer.music.stop()
        self.playing = False
        self.seek_bar.set(0)
        self.current_time_label.config(text="0:00")

    def seek_music(self, value):
        # シークバーの値に応じて再生位置を設定
        if self.playing:
            pygame.mixer.music.rewind()
            pygame.mixer.music.set_pos(float(value))

    def update_seek_bar(self):
        # スレッドで再生位置を追跡しシークバーを更新
        def update():
            while self.playing:
                current_time = pygame.mixer.music.get_pos() / 1000.0  # ミリ秒から秒に変換
                self.seek_bar.set(current_time)
                self.current_time_label.config(text=self.format_time(current_time))
                time.sleep(0.5)
        threading.Thread(target=update, daemon=True).start()

    def format_time(self, seconds):
        # 秒を分:秒の形式にフォーマット
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02}"

# Tkinterアプリケーションの起動
root = tk.Tk()
app = MusicPlayer(root)
root.mainloop()

import tkinter as tk
from tkinter import ttk
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# class Audio(tk.Frame):
#     def __init__(self, master: tk.Tk, props: dict):
#         super().__init__(master)

#         self.props = props
#         self.SR = 16000
#         # テーブルの作成
#         # self.create_audio()

#         # アプリケーション全体で独自イベントを捉える
#         self.master.bind_all("<<OpenFile>>", lambda _: self.open_file())

#     def open_file(self):
#         # ファイルパスの取得
#         file_name = self.props.get("open_file_name")
#         # 音声ファイルの読み込み
#         self.audio_data, _ = librosa.load(file_name, sr=self.SR)

#         self.frame_waveform = tk.Frame(self)
#         fig, ax = plt.subplots()
#         canvas = FigureCanvasTkAgg(fig, master=self.frame_waveform)	# masterに対象とするframeを指定
#         plt.plot(np.arange(len(self.audio_data))/self.SR, self.audio_data)				# 描画データを追加
#         plt.xlabel('time(s)')					# x軸のラベルを設定
#         plt.xticks(np.linspace(0,len(self.audio_data)/self.SR,15))
#         plt.grid()
#         plt.show()
#         canvas.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

#         print(file_name)

class Audio(tk.Frame):
    def __init__(self, master: tk.Tk, props: dict):
        super().__init__(master)
        self.props = props
        self.SR = 16000
        self.master.bind_all("<<OpenFile>>", lambda _: self.open_file())

    def open_file(self):
        file_name = self.props.get("open_file_name")
        if not file_name:
            return
        
        try:
            self.audio_data, _ = librosa.load(file_name, sr=self.SR)
            self.display_waveform()
            print(f"Loaded file: {file_name}")
        except Exception as e:
            print(f"Error loading file: {e}")

    def display_waveform(self):
        # Clear previous waveform if it exists
        # for widget in self.frame_waveform.winfo_children():
        #     widget.destroy()
        
        self.frame_waveform = tk.Frame(self)
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=self.frame_waveform)
        
        # Plot waveform
        ax.plot(np.arange(len(self.audio_data)) / self.SR, self.audio_data)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        
        canvas.draw()
        canvas.get_tk_widget().pack(side="left")
        self.frame_waveform.pack()
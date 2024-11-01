import pygame
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time
import numpy as np
import librosa
import scipy.io.wavfile
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioPlayer:
    TIME_INIT_STR = "--:--.-"
    LV_METER_UPDATE_SEC = 0.3

    def __init__(self, tk_root):
        self.tk_root = tk_root
        self.tk_root.title("Audio Player")

        # self.tk_root.geometry("480x620")
        self.tk_root.resizable(False, False)
        
        self.init_pygame()
        self.create_widgets()

    def init_pygame(self):
        self.audio_file_path = ""
        pygame.mixer.init()
        self.audio_data = None
        self.audio_sr = None
    
    def init_load_audio(self):
        self.audio_file_path = "rec/a.wav"
        if self.audio_file_path:
            self.stop_audio()
            self.load_audio_data(self.audio_file_path)
            pygame.mixer.music.load(self.audio_file_path)
            self.status_label.config(text=f"Loaded: {self.audio_file_path}")
            self.elapsed_time = 0
            self.update_time_label()

    def load_audio(self):
        self.audio_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.audio_file_path:
            self.stop_audio()
            self.load_audio_data(self.audio_file_path)
            pygame.mixer.music.load(self.audio_file_path)
            self.status_label.config(text=f"Loaded: {self.audio_file_path}")
            self.elapsed_time = 0
            self.update_time_label()

    def load_audio_data(self, file_path):
        self.audio_sr = 16000
        self.audio_data, self.audio_sr = librosa.load(file_path, dtype=np.float64, sr=self.audio_sr)
        self.current_audio_data_changed = self.audio_data
        self.duration = len(self.current_audio_data_changed)/self.audio_sr
        self.make_spectrogram()

    def play_audio_default(self):
        if len(self.audio_file_path) > 0:
            pygame.mixer.music.load(self.audio_file_path)
            pygame.mixer.music.play()
            audio_file_name = self.audio_file_path.split('/')[-1]
            self.status_label.config(text=f"Playing... {audio_file_name}")
            self.start_time_counter()
            self.start_level_meter()
            self.start_seek_bar()
            self.start_spac_meter()

    def play_audio(self):
        if len(self.audio_file_path) > 0:
            pygame.mixer.music.load(self.temporary_file)
            pygame.mixer.music.play(loops=0, start=0.0)
            audio_file_name = self.audio_file_path.split('/')[-1]
            self.status_label.config(text=f"Playing... {audio_file_name} changed")
            self.start_time_counter()
            self.start_level_meter()
            self.start_seek_bar()
            self.start_spac_meter()
            # self.start_effector()
        # self.play_audio()

    def stop_audio(self):
        pygame.mixer.music.stop()
        self.elapsed_time = 0
        self.status_label.config(text="Stopped")
        self.time_label.config(text=AudioPlayer.TIME_INIT_STR)
        self.level_meter_canvas.delete("all")

    def start_time_counter(self):
        def count_time():
            self.elapsed_time = 0
            while pygame.mixer.music.get_busy():
                self.update_time_label()
                time.sleep(0.1)
                self.elapsed_time += 1
            self.time_label.config(text=AudioPlayer.TIME_INIT_STR)

        Thread(target=count_time, daemon=True).start()

    def update_time_label(self):
        minutes, seconds = divmod(self.elapsed_time, 600)
        seconds, microseconds = divmod(seconds, 10)
        time_str = f"{minutes:02}:{seconds:02}.{microseconds:01}"
        self.time_label.config(text=time_str)

    def create_widgets(self):
        self.main_frame = tk.Frame(self.tk_root)
        self.main_frame.pack(side="left")

        self.init_effector()

        load_button = tk.Button(self.main_frame, text="Load", command=self.load_audio)
        play_button = tk.Button(self.main_frame, text="Play", command=self.play_audio)
        play_button_default = tk.Button(self.main_frame, text="Play Default", command=self.play_audio_default)
        stop_button = tk.Button(self.main_frame, text="Stop", command=self.stop_audio)
        
        load_button.pack(pady=5)
        play_button.pack(pady=5)
        play_button_default.pack(pady=5)
        stop_button.pack(pady=5)

        self.status_label = tk.Label(self.main_frame, text="No file loaded")
        self.status_label.pack(pady=5)

        self.time_label = tk.Label(self.main_frame, text=AudioPlayer.TIME_INIT_STR, font=("Helvetica", 16))
        self.time_label.pack(pady=5)
        
        self.level_meter_canvas = tk.Canvas(self.main_frame, width=400, height=10, bg="black")
        self.level_meter_canvas.pack(pady=5)

        self.seek_bar_canvas = tk.Canvas(self.main_frame, width=400, height=10, bg="black")
        self.seek_bar_canvas.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5,1.5))
        self.spec_meter_canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)#tk.Canvas(self.main_frame, width=400, height=100, bg="black")
        self.spec_meter_canvas.get_tk_widget().pack(side="top") # self.spec_meter_canvas.pack(pady=5)

        self.init_load_audio()

        self.bottom_frame = tk.Frame(self.tk_root)
        self.bottom_frame.pack(side="bottom")
        self.fig_B, self.ax_B = plt.subplots(figsize=(5,5))
        self.bottom_canvas = FigureCanvasTkAgg(self.fig_B, master=self.bottom_frame)
        self.bottom_canvas.get_tk_widget().pack(side="top")

    def start_level_meter(self):
        def update_level():
            while pygame.mixer.music.get_busy():
                self.draw_level_meter(AudioPlayer.LV_METER_UPDATE_SEC)
                time.sleep(AudioPlayer.LV_METER_UPDATE_SEC)
            self.level_meter_canvas.delete("all")

        Thread(target=update_level, daemon=True).start()

    def draw_level_meter(self, search_len_sec):
        if self.audio_data is None:
            return

        # 現在の再生位置を取得
        start_sec = pygame.mixer.music.get_pos() / 1000.0  # ミリ秒から秒に変換
        start_index = int(start_sec * self.audio_sr)
        end_index = int((start_sec + search_len_sec) * self.audio_sr)

        if end_index > len(self.audio_data):
            end_index = len(self.audio_data)

        current_audio_segment = self.audio_data[start_index:end_index]
        level = np.abs(current_audio_segment).mean()

        if level < 0.5:
            lm_color = "green"
        elif level < 0.8:
            lm_color = "yellow"
        else:
            lm_color = "red"

        self.level_meter_canvas.delete("all")
        self.level_meter_canvas.create_rectangle(0, 0, 400 * level, 50, fill=lm_color)
    
    def start_seek_bar(self):
        def update_level():
            while pygame.mixer.music.get_busy():
                self.draw_seek_bar(AudioPlayer.LV_METER_UPDATE_SEC)
                time.sleep(AudioPlayer.LV_METER_UPDATE_SEC)
            self.seek_bar_canvas.delete("all")

        Thread(target=update_level, daemon=True).start()

    def draw_seek_bar(self, search_len_sec):
        if self.audio_data is None:
            return

        # 現在の再生位置を取得
        start_sec = pygame.mixer.music.get_pos() / 1000.0  # ミリ秒から秒に変換
        sec = start_sec / (len(self.audio_data)/self.audio_sr)
        self.seek_bar_canvas.delete("all")
        self.seek_bar_canvas.create_rectangle(0, 0, 400 * sec, 10, fill="red")

    def make_spectrogram(self):
        self.size_frame = 4096	# フレームサイズ
        size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)
        hamming_window = np.hamming(self.size_frame)     # ハミング窓
        # フレーム毎にスペクトルを計算
        self.spectrogram =[]
        for i in np.arange(0, len(self.current_audio_data_changed)-self.size_frame, size_shift):
            # 該当フレームのデータを取得
            idx = int(i)	# arangeのインデクスはfloatなのでintに変換
            x_frame = self.current_audio_data_changed[idx : idx+self.size_frame]
            fft_spec = np.fft.rfft(x_frame * hamming_window)
            fft_log_abs_spec = np.log(np.abs(fft_spec))
            self.spectrogram.append(fft_log_abs_spec)

    def start_spac_meter(self):
        def update_level():
            while pygame.mixer.music.get_busy():
                self.draw_spec_meter(AudioPlayer.LV_METER_UPDATE_SEC)
                time.sleep(AudioPlayer.LV_METER_UPDATE_SEC)
            self.seek_bar_canvas.delete("all")
        Thread(target=update_level, daemon=True).start()

    def draw_spec_meter(self, search_len_sec):
        if self.audio_data is None:
            return
        
        start_sec = pygame.mixer.music.get_pos() / 1000.0  # ミリ秒から秒に変換

        index = int((len(self.spectrogram)-1) * (start_sec / self.duration))
        fft_spec = self.spectrogram[index]
        
        # 波形データをプロット
        self.ax.cla()
        x_data = np.fft.rfftfreq(256, d=1/self.audio_sr)
        self.ax.bar(x_data, np.abs(fft_spec[:129]), width=0.8)
        self.ax.set_ylim(0, 5)
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylabel('amblitude')
        self.ax.set_xlabel('frequency [Hz]')
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.spec_meter_canvas.draw()

    def choice(self):
        if self.voice_change.get() : self.label.config(text="vc")
        else: self.label.config(text="aa")
    
    def voice_changer(self, val):
        self.voice_changer_freq = val
        self.lavel_voice_change.config(text="R="+str(self.voice_changer_freq))

    def tremolo_er(self, val):
        self.tremolo_R = self.scale_tremolo_R.get()
        self.tremolo_D = self.scale_tremolo_D.get()
        self.lavel_tremolo_R.config(text="R="+str(self.tremolo_R))
        self.lavel_tremolo_D.config(text="D="+str(self.tremolo_D))

    def vibrato_er(self, val):
        self.vibrato_R = self.scale_vibrato_R.get()
        self.vibrato_D = self.scale_vibrato_D.get()
        self.lavel_vibrato_R.config(text="R="+str(self.vibrato_R))
        self.lavel_vibrato_D.config(text="D="+str(self.vibrato_D))

    def echo_er(self, val):
        self.echo_Delay = self.scale_echo_Delay.get()
        self.echo_Vol = self.scale_echo_Vol.get()
        self.label_echo_Delay.config(text="Delay="+str(self.echo_Delay)+"sec")
        self.label_echo_Vol.config(text="Volume="+str(self.echo_Vol))
    
    def compressor_er(self,val):
        self.compressor_r = val
        self.label_compressor.config(text="threshold="+str(self.compressor_r))

    def effector_clear(self):
        if self.audio_file_path:
            self.audio_data, self.audio_sr = librosa.load(self.audio_file_path, dtype=np.float64)
            self.current_audio_data_changed = self.audio_data
            with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', dir='rec/temp') as temp_file:
                self.temporary_file = temp_file.name
            scipy.io.wavfile.write(self.temporary_file, int(self.audio_sr), self.audio_data)

    def init_effector(self):
        self.temporary_file = 'rec/temporary_file.wav'

        self.frame_voice_change = tk.Frame(self.main_frame)
        self.frame_voice_change.pack(side="right")
        self.frame_tremolo = tk.Frame(self.frame_voice_change)
        self.frame_tremolo.pack(side="right")
        self.frame_vibrato = tk.Frame(self.frame_tremolo)
        self.frame_vibrato.pack(side="right")
        self.frame_echo = tk.Frame(self.frame_vibrato)
        self.frame_echo.pack(side="right")
        self.frame_compressor = tk.Frame(self.frame_echo)
        self.frame_compressor.pack(side="right")

        # self.voice_change = tk.BooleanVar()
        # self.tremolo = tk.BooleanVar()
        # self.vibrato = tk.BooleanVar()
        # tk.Checkbutton(self.frame_voice_change, variable=self.voice_change, text="voice_change").pack()
        # tk.Checkbutton(self.frame_tremolo, variable=self.tremolo,      text="tremolo").pack()
        # tk.Checkbutton(self.frame_vibrato, variable=self.vibrato,      text="vibrato").pack()

        tk.Button(self.frame_voice_change,text="voice change", command=self.apply_voice_change).pack(pady=5)
        tk.Button(self.frame_tremolo,text="tremolo", command=self.apply_toremolo).pack(pady=5)
        tk.Button(self.frame_vibrato,text="vibrato", command=self.apply_vibrato).pack(pady=5)
        tk.Button(self.frame_echo,text="echo", command=self.apply_echo).pack(pady=5)
        tk.Button(self.frame_compressor,text="compressor", command=self.apply_compressor).pack(pady=5)

        # Voice Change
        self.lavel_voice_change = tk.Label(self.frame_voice_change, text='R=0')
        self.lavel_voice_change.pack(side="top")
        self.scale_voice_change = tk.Scale(
            command=self.voice_changer,
            master=self.frame_voice_change,				# 表示するフレーム
            from_=0,					# 最小値
            to=2000,     				# 最大値
            resolution=1.0,	            # 刻み幅
            label=u'Frequency [Hz]',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=400,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_voice_change.pack(side="top")

        # tremolo
        self.frame_tremolo_R = tk.Frame(self.frame_tremolo)
        self.frame_tremolo_R.pack(side="left")
        self.lavel_tremolo_R = tk.Label(self.frame_tremolo_R, text='R=0')
        self.lavel_tremolo_R.pack(side="top")
        self.scale_tremolo_R = tk.Scale(
            command=self.tremolo_er,
            master=self.frame_tremolo_R,				# 表示するフレーム
            from_=0,					# 最小値
            to=10,     				# 最大値
            resolution=0.01,	            # 刻み幅
            label=u'R',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_tremolo_R.pack(side="top")

        self.frame_tremolo_D = tk.Frame(self.frame_tremolo)
        self.frame_tremolo_D.pack(side="right")
        self.lavel_tremolo_D = tk.Label(self.frame_tremolo_D, text='R=0')
        self.lavel_tremolo_D.pack()
        self.scale_tremolo_D = tk.Scale(
            command=self.tremolo_er,
            master=self.frame_tremolo_D,				# 表示するフレーム
            from_=0,					# 最小値
            to=5,     				# 最大値
            resolution=0.01,	            # 刻み幅
            label=u'D',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_tremolo_D.pack(side="top")

        # vibrato
        self.frame_vibrato_R = tk.Frame(self.frame_vibrato)
        self.frame_vibrato_R.pack(side="left")
        self.lavel_vibrato_R = tk.Label(self.frame_vibrato_R, text='R=0')
        self.lavel_vibrato_R.pack(side="top")
        self.scale_vibrato_R = tk.Scale(
            command=self.vibrato_er,
            master=self.frame_vibrato_R,				# 表示するフレーム
            from_=0,					# 最小値
            to=10,     				# 最大値
            resolution=0.01,	            # 刻み幅
            label=u'R',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_vibrato_R.pack(side="top")

        self.frame_vibrato_D = tk.Frame(self.frame_vibrato)
        self.frame_vibrato_D.pack(side="right")
        self.lavel_vibrato_D = tk.Label(self.frame_vibrato_D, text='R=0')
        self.lavel_vibrato_D.pack()
        self.scale_vibrato_D = tk.Scale(
            command=self.vibrato_er,
            master=self.frame_vibrato_D,				# 表示するフレーム
            from_=0,					# 最小値
            to=300,     				# 最大値
            resolution=0.1,	            # 刻み幅
            label=u'D',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_vibrato_D.pack(side="top")
        
        # echo
        self.frame_echo_Delay = tk.Frame(self.frame_echo)
        self.frame_echo_Delay.pack(side="left")
        self.label_echo_Delay = tk.Label(self.frame_echo_Delay, text='Delay=0')
        self.label_echo_Delay.pack(side="top")
        self.scale_echo_Delay = tk.Scale(
            command=self.echo_er,
            master=self.frame_echo_Delay,# 表示するフレーム
            from_=0,					# 最小値
            to=1,     				# 最大値
            resolution=0.01,	            # 刻み幅
            label=u'Delay',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_echo_Delay.pack(side="top")

        self.frame_echo_Vol = tk.Frame(self.frame_echo)
        self.frame_echo_Vol.pack(side="right")
        self.label_echo_Vol = tk.Label(self.frame_echo_Vol, text='Vol=0')
        self.label_echo_Vol.pack()
        self.scale_echo_Vol = tk.Scale(
            command=self.echo_er,
            master=self.frame_echo_Vol,				# 表示するフレーム
            from_=0,					# 最小値
            to=2,     				# 最大値
            resolution=0.01,	            # 刻み幅
            label=u'Volume',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_echo_Vol.pack(side="top")

        #compressor
        self.frame_compressor = tk.Frame(self.frame_compressor)
        self.frame_compressor.pack(side="right")
        self.label_compressor = tk.Label(self.frame_compressor, text='threshold=0')
        self.label_compressor.pack()
        self.scale_compressor = tk.Scale(
            command=self.compressor_er,
            master=self.frame_compressor,				# 表示するフレーム
            from_=0,					# 最小値
            to=1,     				# 最大値
            resolution=0.01,	            # 刻み幅
            label=u'threshold(%)',
            orient=tk.VERTICAL,	# 横方向にスライド
            length=300,					# 横サイズ
            width=10,					# 縦サイズ
            font=("", 10)				# フォントサイズは20pxに設定
        )
        self.scale_compressor.pack(side="top")
        # 選択されたオプションを格納するための変数
        self.selected_ratio = tk.IntVar(value=1)
        # ラジオボタンの作成
        radiobutton01 = tk.Radiobutton(self.frame_compressor, text="4:1", variable=self.selected_ratio, value=4)
        radiobutton01.pack()
        radiobutton02 = tk.Radiobutton(self.frame_compressor, text="6:1", variable=self.selected_ratio, value=6)
        radiobutton02.pack()
        radiobutton03 = tk.Radiobutton(self.frame_compressor, text="10:1", variable=self.selected_ratio, value=10)
        radiobutton03.pack()
        radiobutton04 = tk.Radiobutton(self.frame_compressor, text="limiter", variable=self.selected_ratio, value=1000)
        radiobutton04.pack()

        # self.entry = tk.Entry(self.frame_voice_change, width=20)
        # self.entry.pack(pady=10)
        self.label = tk.Label(self.main_frame, text='出力')
        self.label.pack()
        tk.Button(self.main_frame,text="effecta", command=self.choice).pack(pady=5)

        tk.Button(self.main_frame,text="effector clear", command=self.effector_clear).pack(pady=5)

    def generate_sinusoid(self, sampling_rate, frequency, duration):
        sampling_interval = 1.0 / sampling_rate
        t = np.arange(sampling_rate * duration) * sampling_interval
        waveform = np.sin(2.0 * np.pi * frequency * t)
        return waveform
    
    def apply_voice_change(self):
        if self.current_audio_data_changed is None:
            return

        #voice_change
        frequency = self.scale_voice_change.get()  # 生成する正弦波の時間的長さ
        duration = len(self.current_audio_data_changed)
        sin_wave = self.generate_sinusoid(self.audio_sr, frequency, duration/self.audio_sr)
        sin_wave = sin_wave * 0.9   # 最大値を0.9にする
        audio_data_changed = self.current_audio_data_changed * sin_wave[:len(self.current_audio_data_changed)]         # 元の音声と正弦波を重ね合わせる

        self.current_audio_data_changed = audio_data_changed

        audio_data_changed = (audio_data_changed * 32768.0). astype('int16')  # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
        # 音声ファイルとして出力する
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', dir='rec/temp') as temp_file:
            self.temporary_file = temp_file.name
        scipy.io.wavfile.write(self.temporary_file, int(self.audio_sr), audio_data_changed)

    def apply_toremolo(self):
        if self.current_audio_data_changed is None:
            return

        #tremolo
        self.tremolo_R = self.scale_tremolo_R.get()
        self.tremolo_D = self.scale_tremolo_D.get()
        f_s = 1
        frequency = self.tremolo_R / f_s                                                  # 生成する正弦波の周波数（Hz）
        duration = len(self.current_audio_data_changed)                                                   # 生成する正弦波の時間的長さ
        sin_wave = self.generate_sinusoid(self.audio_sr, frequency, duration/self.audio_sr)     # 正弦波を生成する
        sin_wave = ((1 + self.tremolo_D * sin_wave)/(1+self.tremolo_D)) * 0.9       # 最大値を0.9にする
        audio_data_changed = self.current_audio_data_changed * sin_wave[:len(self.current_audio_data_changed)]         # 元の音声と正弦波を重ね合わせる

        self.current_audio_data_changed = audio_data_changed

        audio_data_changed = (audio_data_changed * 32768.0). astype('int16')  # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
        # 音声ファイルとして出力する
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', dir='rec/temp') as temp_file:
            self.temporary_file = temp_file.name
        scipy.io.wavfile.write(self.temporary_file, int(self.audio_sr), audio_data_changed)
    
    def apply_vibrato(self):
        if self.current_audio_data_changed is None:
            return

        #vibrato
        self.vibrato_R = self.scale_vibrato_R.get()
        self.vibrato_D = self.scale_vibrato_D.get()
        f_s = 1
        frequency = self.vibrato_R / f_s                                                  # 生成する正弦波の周波数（Hz）
        duration = len(self.current_audio_data_changed)     
        sin_wave = self.generate_sinusoid(self.audio_sr, frequency, duration/self.audio_sr)     # 正弦波を生成する
        
        # 最大値を0.9にする
        tau = self.vibrato_D * sin_wave

        # 音声波形を時間軸上で前後で揺らす
        audio_data_changed = np.zeros_like(self.current_audio_data_changed)
        for t in np.arange(len(self.current_audio_data_changed)-1):
            if(t-int(tau[t]) > 0 and t-int(tau[t]) < len(self.current_audio_data_changed)-2):
                audio_data_changed[t] = self.current_audio_data_changed[t-int(tau[t])]
            else:
                audio_data_changed[t] = self.current_audio_data_changed[t]
        
        self.current_audio_data_changed = audio_data_changed

        audio_data_changed = (audio_data_changed * 32768.0). astype('int16')  # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する

        # 音声ファイルとして出力する
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', dir='rec/temp') as temp_file:
            self.temporary_file = temp_file.name
        scipy.io.wavfile.write(self.temporary_file, int(self.audio_sr), audio_data_changed)

    def apply_echo(self):
        delay_samples = int(self.audio_sr * self.scale_echo_Delay.get())
        echo_audio = np.zeros(len(self.current_audio_data_changed) + delay_samples)
        echo_audio[:len(self.audio_data)] += self.current_audio_data_changed
        echo_audio[delay_samples:delay_samples + len(self.current_audio_data_changed)] += self.scale_echo_Vol.get() * self.current_audio_data_changed

        echo_audio = echo_audio[:len(self.current_audio_data_changed)]
        self.current_audio_data_changed = echo_audio

        echo_audio = (self.current_audio_data_changed * 32768.0). astype('int16')  # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する

        # 音声ファイルとして出力する
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', dir='rec/temp') as temp_file:
            self.temporary_file = temp_file.name
        scipy.io.wavfile.write(self.temporary_file, int(self.audio_sr), echo_audio)    

    def apply_compressor(self):
        threshold = self.scale_compressor.get() * max(self.current_audio_data_changed)
        ratio = self.selected_ratio.get()

        compressed_audio = np.copy(self.current_audio_data_changed)
        above_threshold = np.abs(compressed_audio) > threshold
        compressed_audio[above_threshold] = threshold + (compressed_audio[above_threshold] - threshold) / ratio        
        self.current_audio_data_changed = compressed_audio

        compressed_audio = (compressed_audio * 32768.0). astype('int16')  # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する

        # 音声ファイルとして出力する
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav', dir='rec/temp') as temp_file:
            self.temporary_file = temp_file.name
        scipy.io.wavfile.write(self.temporary_file, int(self.audio_sr), compressed_audio)

        size_frame = 4096
        size_shift = 100
        volume = []
        for i in np.arange(0, len(compressed_audio)-size_frame, size_shift):
	
            # 該当フレームのデータを取得
            idx = int(i)	# arangeのインデクスはfloatなのでintに変換
            x_frame = compressed_audio[idx : idx+size_frame]
            # 音量
            vol = 20 * np.log10(np.mean(x_frame ** 2) + 0.01)
            volume.append(vol)

        # 波形データをプロット
        self.ax_B.cla()
        x_data = np.linspace(0, self.duration, len(volume))
        self.ax_B.plot(x_data, volume)
        self.ax_B.set_ylabel('volume [dB]')
        self.ax_B.set_xlabel('time [sec]')
        self.ax_B.set_title("compressor volume")
        self.bottom_canvas.draw()

    def start_effector(self):
        def update_level():
            while pygame.mixer.music.get_busy():
                self.apply_effector(AudioPlayer.LV_METER_UPDATE_SEC)
                time.sleep(AudioPlayer.LV_METER_UPDATE_SEC)
        Thread(target=update_level, daemon=True).start()

    def apply_effector(self):
        if self.audio_data is None:
            return

        # 現在の再生位置を取得
        # start_sec = pygame.mixer.music.get_pos() / 1000.0  # ミリ秒から秒に変換
        # start_index = int(start_sec * self.audio_sr)
        # end_index = int((start_sec + search_len_sec) * self.audio_sr)

        # if end_index > len(self.audio_data):
        #     end_index = len(self.audio_data)

        # current_audio_segment = self.audio_data[start_index:end_index]
        
        current_audio_segment = self.audio_data

        #voice_change
        frequency = 200.0

        # 生成する正弦波の時間的長さ
        duration = len(current_audio_segment) 
        sin_wave = self.generate_sinusoid(self.audio_sr, frequency, duration/self.audio_sr)
        sin_wave = sin_wave * 0.9   # 最大値を0.9にする

        # 元の音声と正弦波を重ね合わせる
        self.current_audio_segment_changed = current_audio_segment * sin_wave

        #tremolo
        # f_s = 1
        # frequency = self.tremolo_R / f_s                                                  # 生成する正弦波の周波数（Hz）
        # duration = len(current_audio_segment)                                                   # 生成する正弦波の時間的長さ
        # sin_wave = self.generate_sinusoid(self.audio_sr, frequency, duration/self.audio_sr)     # 正弦波を生成する
        # sin_wave = ((1 + self.tremolo_D * sin_wave)/(1+self.tremolo_D)) * 0.9       # 最大値を0.9にする
        # current_audio_segment_changed = current_audio_segment * sin_wave                        # 元の音声と正弦波を重ね合わせる

        # current_audio_segment_changed = (current_audio_segment_changed * 32768.0). astype('int16')          # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する


if __name__ == '__main__':
    tk_root = tk.Tk()
    app = AudioPlayer(tk_root)
    tk_root.mainloop()



# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

def plot_spectrogram(current_audio_data_changed,ax,canvas):
    size_frame = 4096	# フレームサイズ
    size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)
    hamming_window = np.hamming(size_frame)     # ハミング窓
    SR = 16000
    # フレーム毎にスペクトルを計算
    spectrogram =[]
    for i in np.arange(0, len(current_audio_data_changed)-size_frame, size_shift):
        # 該当フレームのデータを取得
        idx = int(i)	# arangeのインデクスはfloatなのでintに変換
        x_frame = current_audio_data_changed[idx : idx + size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec[:256])

    ax.cla()
    ax.imshow(
        np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転置して上下反転
        extent=[0, len(current_audio_data_changed)/SR, 0, SR/(2*8)],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
        aspect='auto',
        interpolation='nearest'
    )
    ax.set_ylabel('time [s]')
    ax.set_ylabel('frequency [Hz]')
    canvas.draw()




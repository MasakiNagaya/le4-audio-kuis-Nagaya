#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，フーリエ変換を行う．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

name1 = "a_train"
name2 = "voice_change"
name3 = "voice_change_100"
name4 = "voice_change_400"
name5 = "voice_change_800"
name6 = "voice_change_1600"

# 音声ファイルの読み込み
x, _ = librosa.load('rec/'+name1+'.wav', sr=SR)
fft_spec = np.fft.rfft(x)
fft_log_abs_spec = np.log(np.abs(fft_spec))

# 音声ファイルの読み込み
vc2, _ = librosa.load('rec/'+name2+'.wav', sr=SR)
fft_spec_vc = np.fft.rfft(vc2)
fft_log_abs_spec_vc2 = np.log(np.abs(fft_spec_vc))

# 音声ファイルの読み込み
vc3, _ = librosa.load('rec/'+name3+'.wav', sr=SR)
fft_spec_vc = np.fft.rfft(vc3)
fft_log_abs_spec_vc3 = np.log(np.abs(fft_spec_vc))

# 音声ファイルの読み込み
vc4, _ = librosa.load('rec/'+name4+'.wav', sr=SR)
fft_spec_vc = np.fft.rfft(vc4)
fft_log_abs_spec_vc4 = np.log(np.abs(fft_spec_vc))

# 音声ファイルの読み込み
vc5, _ = librosa.load('rec/'+name5+'.wav', sr=SR)
fft_spec_vc = np.fft.rfft(vc5)
fft_log_abs_spec_vc5 = np.log(np.abs(fft_spec_vc))

# 音声ファイルの読み込み
vc6, _ = librosa.load('rec/'+name6+'.wav', sr=SR)
fft_spec_vc = np.fft.rfft(vc6)
fft_log_abs_spec_vc6 = np.log(np.abs(fft_spec_vc))

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定
# x軸のデータを生成（元々のデータが0~8000Hzに対応するようにする）
#x_data = np.linspace((SR/2)/len(fft_log_abs_spec), SR/2, len(fft_log_abs_spec))
x_data = np.fft.rfftfreq(len(x), d=1/SR)
plt.plot(x_data, fft_log_abs_spec, alpha=0.7, label="default")			# 描画
plt.plot(x_data, fft_log_abs_spec_vc2, alpha=0.7, label="voice change 200Hz")			# 描画
plt.plot(x_data, fft_log_abs_spec_vc3, alpha=0.7, label="voice change 100Hz")
plt.plot(x_data, fft_log_abs_spec_vc4, alpha=0.7, label="voice change 400Hz")
plt.plot(x_data, fft_log_abs_spec_vc5, alpha=0.7, label="voice change 800Hz")
plt.plot(x_data, fft_log_abs_spec_vc6, alpha=0.7, label="voice change 1600Hz")
plt.grid()
plt.legend()
# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-spectrum-whole_vc_ex18.png')

# 横軸を0~2000Hzに拡大
# xlimで表示の領域を変えるだけ
fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.xlim([0, 2000])
plt.plot(x_data, fft_log_abs_spec, alpha=0.7, label="default")
plt.plot(x_data, fft_log_abs_spec_vc2, alpha=0.7, label="voice change 200Hz")			# 描画
plt.plot(x_data, fft_log_abs_spec_vc3, alpha=0.7, label="voice change 100Hz")
plt.plot(x_data, fft_log_abs_spec_vc4, alpha=0.7, label="voice change 400Hz")
plt.plot(x_data, fft_log_abs_spec_vc5, alpha=0.7, label="voice change 800Hz")
plt.plot(x_data, fft_log_abs_spec_vc6, alpha=0.7, label="voice change 1600Hz")
plt.grid()
plt.legend()
# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-spectrum-2000_vc_ex18.png')

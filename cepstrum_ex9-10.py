#ex9,10

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	# 1. 振幅スペクトルの対数をとる．
	log_spectrum = np.log(amplitude_spectrum)
	# 2. 対数振幅スペクトルをフーリエ変換する．
	cepstrum = np.fft.rfft(log_spectrum)
	return cepstrum

# サンプリングレート
SR = 16000

name = "a"

# 音声ファイルの読み込み
x1, _ = librosa.load('rec/'+name+'1.wav', sr=SR)
x2, _ = librosa.load('rec/'+name+'2.wav', sr=SR)

def ifft_cepstrum(x):
    # 高速フーリエ変換
    # np.fft.rfftを使用するとFFTの前半部分のみが得られる
    fft_spec = np.fft.rfft(x)

    cepstrum_spec = cepstrum(fft_spec)

    # 3. フーリエ変換の結果のうち，低い周波数の成分のみを取り出す．
    cepstrum_spec_low = np.zeros_like(cepstrum_spec)
    # cp_len = len(cepstrum_spec_low) - 1
    for i in range(0,13):
        cepstrum_spec_low[i] = cepstrum_spec[i]
        # cepstrum_spec_low[cp_len - i] = cepstrum_spec[cp_len - i]
    # 4. 取り出した成分のみを逆フーリエ変換する．
    fft_spec_low = np.fft.irfft(cepstrum_spec_low)

    log_spectrum_all = np.log(fft_spec)
    fft_spec_low = np.append(fft_spec_low,-2)

    return log_spectrum_all, fft_spec_low

log_spectrum_all1, fft_spec_low1 = ifft_cepstrum(x1)
log_spectrum_all2, fft_spec_low2 = ifft_cepstrum(x2)

#
# スペクトルを画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定
# x軸のデータを生成（元々のデータが0~8000Hzに対応するようにする）
#x_data = np.linspace((SR/2)/len(fft_log_abs_spec), SR/2, len(fft_log_abs_spec))
x_data1 = np.fft.rfftfreq(len(x1), d=1/SR)
x_data2 = np.fft.rfftfreq(len(x2), d=1/SR)
# print(len(x_data))
plt.plot(x_data1, log_spectrum_all1, alpha=0.7, label = 'smooth')			# 描画
plt.plot(x_data2, log_spectrum_all2, alpha=0.7, label = 'separate')			# 描画
plt.plot(x_data1, fft_spec_low1, label = 'smooth')			# 描画
plt.plot(x_data2, fft_spec_low2, label = 'separate')			# 描画
plt.legend()
# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-spectrum-whole_'+name+'.png')

# 横軸を0~2000Hzに拡大
# xlimで表示の領域を変えるだけ
fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.xlim([0, 2000])
plt.plot(x_data1, log_spectrum_all1, alpha=0.7, label = 'smooth')			# 描画
plt.plot(x_data2, log_spectrum_all2, alpha=0.7, label = 'separate')			# 描画
plt.plot(x_data1, fft_spec_low1, label = 'smooth')			# 描画
plt.plot(x_data2, fft_spec_low2, label = 'separate')			# 描画
plt.legend()
# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-spectrum-2000_'+name+'.png')
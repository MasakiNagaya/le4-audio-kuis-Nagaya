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
import time

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	if a[index-1] < a[index] and a[index+1] < a[index]:
		return True
	else: 
		return False

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('rec/aiueo2.wav', sr=SR)
# print(len(x))
# 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る

start = time.perf_counter() #計測開始
autocorr = np.correlate(x, x, 'full')
autocorr = autocorr [len (autocorr ) // 2 : ]
end = time.perf_counter() #計測終了
# print(len(autocorr))
print('numpy correlate'+'{:.10f}'.format((end-start)/60)) # 87.97(秒→分に直し、小数点以下の桁数を指定して出力)

#fft^2 ifft
start = time.perf_counter() #計測開始
fft_spec = np.fft.rfft(x)
selfcorr = np.fft.irfft(fft_spec ** 2)
end = time.perf_counter() #計測終了
# print(len(selfcorr))
print('self correlate'+'{:.10f}'.format((end-start)/60)) # 87.97(秒→分に直し、小数点以下の桁数を指定して出力)

#
# 自己相関を画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# 音量を描画
plt.xlabel('tau(s)')					# x軸のラベルを設定
plt.ylabel('corr')		# y軸のラベルを設定
plt.plot(autocorr)
plt.plot(selfcorr, alpha=0.7)
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-corr-extra3-aiueo2-r.png')
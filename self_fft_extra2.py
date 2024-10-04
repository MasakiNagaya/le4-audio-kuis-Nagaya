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

im = 1j

# start = time.perf_counter() #計測開始
# ##################################
# #####                        #####
# #####    計測したい処理を実行    #####
# #####                        #####
# #################################
# end = time.perf_counter() #計測終了
# print('{:.2f}'.format((end-start)/60)) # 87.97(秒→分に直し、小数点以下の桁数を指定して出力)

# サンプリングレート
SR = 16000

name = "aiueo1"

# 音声ファイルの読み込み
x, _ = librosa.load('rec/'+name+'.wav', sr=SR)
prelen = len(x)

# 高速フーリエ変換
# np.fft.rfftを使用するとFFTの前半部分のみが得られる
start = time.perf_counter() #計測開始
fft_spec = np.fft.rfft(x)
end = time.perf_counter() #計測終了

print('numpy rfft'+'{:.10f}'.format((end-start)/60)) # 87.97(秒→分に直し、小数点以下の桁数を指定して出力)

#self 高速フーリエ変換
def myFFT(realVec,n):
    if n == 1:
        return realVec # N = 1 のときは入力 = 出力
    else:
        fourierVec = np.zeros(n, dtype=np.complex128) # 成分が複素数のため
 
        # 偶数行成分
        nextEvenInput = realVec[0:n//2] + realVec[n//2:n] 
        fourierVec[0:n:2] = myFFT(nextEvenInput,n//2)
 
        # 奇数行成分
        exp_part = -1 * np.array(np.array(range(n//2))) * 2 / n * np.pi * 1j
        w_factor = np.exp(exp_part)
        nextOddInput = w_factor * (realVec[0:n//2] - realVec[n//2:n])
        fourierVec[1:n:2] = myFFT(nextOddInput,n//2)
        
        return fourierVec

start = time.perf_counter() #計測開始
#要素数拡大
num = int(np.log2(len(x)))
while(len(x)<np.exp2(num+1)):
    x = np.append(x,0)
    
fourierVec = myFFT(x,len(x))
fft_spec_self = fourierVec[:(len(fourierVec)//2)]

end = time.perf_counter() #計測終了
print('numpy selffft'+'{:.10f}'.format((end-start)/60)) # 87.97(秒→分に直し、小数点以下の桁数を指定して出力)

# def myDFT(x):
#     N = len(x)
#     W_n = np.exp(1j*2*np.pi/N)
#     W = [[W_n**(i+j) for j in range(N)] for i in range(N)]
#     return W @ x
# start = time.perf_counter() #計測開始
# dft = myDFT(x)
# end = time.perf_counter() #計測終了
# print('numpy selffft'+'{:.10f}'.format((end-start)/60))

# 複素スペクトルを対数振幅スペクトルに
fft_log_abs_spec = np.log(np.abs(fft_spec))
fft_log_abs_spec_self = np.log(np.abs(fft_spec_self))

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
x_data = np.fft.rfftfreq(prelen, d=1/SR)
plt.plot(x_data, fft_log_abs_spec)			# 描画
x_data1 = np.fft.rfftfreq(len(fft_spec_self)*2-1, d=1/SR)
plt.plot(x_data1, fft_log_abs_spec_self, alpha=0.7, linestyle = "dotted")			# 描画
# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-spectrum-whole-self_'+name+'.png')

# 横軸を0~2000Hzに拡大
# xlimで表示の領域を変えるだけ
fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.xlim([0, 2000])
plt.plot(x_data, fft_log_abs_spec)
plt.plot(x_data1, fft_log_abs_spec_self, alpha=0.7, linestyle = "dotted")

# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-spectrum-2000-self_'+name+'.png')

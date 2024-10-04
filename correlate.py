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

# フレームサイズ
size_frame = 4096		# 2のべき乗
# シフトサイズ
size_shift = 16000 / 1000	# 0.01 秒 (10 msec)
# 基本周波数を保存するlist
omega = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for j in np.arange(0, len(x)-size_frame, size_shift):
	# 該当フレームのデータを取得
	idx = int(j)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]

	# 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
	autocorr = np.correlate(x_frame, x_frame, 'full')

	# 不要な前半を捨てる
	autocorr = autocorr [len (autocorr ) // 2 : ]

	# ピークのインデックスを抽出する
	peakindices = [i for i in range (len (autocorr)-1) if is_peak (autocorr, i)]

	# インデックス0 がピークに含まれていれば捨てる
	peakindices = [i for i in peakindices if i != 0]

	# 自己相関が最大となるインデックスを得る
	if peakindices!=[]:
		max_peak_index = max(peakindices , key=lambda index: autocorr [index])

		# インデックスに対応する周波数を得る
		# （自分で実装すること）
		tau = max_peak_index/SR
		
		omega.append(1/tau)

#
# 基本周波数を画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# 音量を描画
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('omega[Hz]')		# y軸のラベルを設定
plt.plot(np.linspace(0,(len(x)-size_frame)/16000, len(omega)), omega)
plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-omega_ex8_aiueo2.png')
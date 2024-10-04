#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，スペクトログラムを計算して図示する．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('rec/aiueo2.wav', sr=SR)

#
# 短時間フーリエ変換
#

# フレームサイズ
size_frame = 512			# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# 音量 loudnessを保存するlist
loudness = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
flag=0
b=-25

for i in np.arange(0, len(x)-size_frame, size_shift):
	#演習5 2種類の「あいうえお」をフレームに分割し，各フレームの音量を図示せよ．横軸を時間，縦軸を音量（単位は dB）とすること．

	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]
	
	#二乗平均
	rms = np.sqrt(np.sum(x_frame ** 2) / len(x_frame))

	# vol
	vol_dB = 20 * np.log10(np.abs(rms))

	if vol_dB > b and flag==0:
		print("start:" + str(i/16000) + "s")
		flag=1
	elif vol_dB < b and flag==1:
		print("end:" + str(i/16000) + "s")
		flag=0
	
	# 計算した音量 loudnessを配列に保存
	loudness.append(vol_dB)

#
# 音量を画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# 音量を描画
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.ylabel('loudness [dB]')		# y軸のラベルを設定
plt.plot(np.linspace(0,(len(x)-size_frame)/16000, len(loudness)),loudness)
loudness_pd = pd.DataFrame(loudness)
plt.scatter(loudness_pd[loudness_pd[0] > -25].index/(100), loudness_pd[loudness_pd[0] > -25][0], color="r")

plt.show()

# 画像ファイルに保存
fig.savefig('picture/plot-loudness_ex6_aiueo2.png')



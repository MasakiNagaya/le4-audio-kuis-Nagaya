#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，波形を図示する．

# ライブラリの読み込み
import matplotlib.pyplot as plt
import librosa
import numpy as np

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('rec/aiueo_01_train.wav', sr=SR)

# xに波形データが保存される
# 第二戻り値はサンプリングレート（ここでは必要ないので _ としている）

# 波形データを標準出力して確認
print(x)

#
# 波形を画像に表示・保存
#

# 画像として保存するための設定
# 画像サイズを 1000 x 400 に設定
fig = plt.figure(figsize=(10, 4))

# 波形を描画
plt.plot(np.arange(len(x))/SR,x)				# 描画データを追加
plt.xlabel('time(s)')					# x軸のラベルを設定
plt.xticks(np.linspace(0,len(x)/SR,15))
plt.grid()
plt.show()										# 表示

# 画像ファイルに保存
fig.savefig('picture/aiueo_01_train.png')



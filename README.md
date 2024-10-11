# le4-audio-kuis

京都大学 工学部 情報学科 計算機科学実験及演習4 音響信号処理 のサンプルソースコードです。

簡易カラオケシステムを作成します。

仮想環境が構築されたあとは，この環境を使用するように，コマンドプロンプトを起
動するたびに下記のコマンドを毎回実行する．
    activate exp4-audio
その後は通常通り python を実行することができる．
「sox FAIL sox: Sorry, ...」と表示された場合は，「set AUDIODRIVER=waveaudio」とコマンドプロンプト上で実行したあとに，
sox コマンドを実行する．

2024/10/11
演習11
：omega（基本周波数）の6倍以上11倍以下で、基本周波数が400以上は切り捨てた

2024/10/10
発展課題３
自己相関はパワースペクトル（振幅スペクトルの 2 乗）の逆フーリエ変換で計算できる．これを証明せよ．また，このように自己相関を計算することの利点について述べよ．さらに，この方法を実装し，numpy.correlate() を用いる場合と実行時間を比較せよ．
利点：速い。fftで可能。
numpy correlate 0.0533864333
self correlate  0.0001047283 
演習７　済
：フレームサイズを4096にして、シフトサイズを160にする。そうすると、4Hz刻みくらいで細かく見ることができる。
演習8　済
発展課題３　済
演習9　済
：rfftで変換したら　irfftで戻さないとダメ
演習10　済　カラーコードが大変

2024/10/4
演習１－６
発展課題１，２　済
   

# le4-audio-kuis

京都大学 工学部 情報学科 計算機科学実験及演習4 音響信号処理 のサンプルソースコードです。

簡易カラオケシステムを作成します。

仮想環境が構築されたあとは，この環境を使用するように，コマンドプロンプトを起
動するたびに下記のコマンドを毎回実行する．
    activate exp4-audio
その後は通常通り python を実行することができる．

2024/10/14
演習１－６
発展課題１，２

途中　発展課題３
自己相関はパワースペクトル（振幅スペクトルの 2 乗）の逆フーリエ変換で計算できる．これを証明せよ．また，このように自己相関を計算することの利点について述べよ．さらに，この方法を実装し，numpy.correlate() を用いる場合と実行時間を比較せよ．
利点：速い。fftで可能。

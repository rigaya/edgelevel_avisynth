﻿
# エッジレベル調整 (Avisynth版)

[がらくたハウスのがらくた置き場様](http://www.geocities.jp/flash3kyuu/)の
エッジレベル調整(edgelevelMT.auf)をAvisynthに移植したものです。

AVX2 / SSE2 で高速化されています。環境に合わせて、最速のものが自動的に選択されます。


## ダウンロード & 更新履歴
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-11.html)

## 想定動作環境

Win 10/11 (x64)  
Avisynth+ (x86/x64)

## 注意事項

無保証です。自己責任で使用してください。  
このフィルタを使用したことによる、いかなる損害・トラブルについても責任を負いません。

## ライセンス

エッジレベル調整のソースコードのライセンスはMITライセンスとしております。

## オプション

### edgelevel(clip, int "strength", int "threshold", int "bc", int "wc", int "thread", int "simd")

- strength (-31 ～ 31, デフォルト 10)  
  エッジの強調度合いを調整します。
  プラスにするとエッジが強調され、マイナスにするとぼけます。

- threshold (0 ～ 255, デフォルト 16)  
 ノイズを無視する閾値です。

- bc (0 ～ 31, デフォルト 0)  
  フィルタ処理の際には、基本的に元の明るさを維持します。
  シュートをつけて輪郭線を意図的に暗くしたいときに使います。
  明るさの変化は局所的です。
  重み付けを調整したので、単純な色の境界でのシュートを出にくくなり、
  アニメなどの輪郭線にのみより強くかかるようになっています。

- wc (0 ～ 31, デフォルト 0)  
   黒補正とは逆の処理です。
   
- thread (0 ～ 32, デフォルト 0 (自動))  
   並列スレッド数です。

- simd (0 ～ 2, デフォルト 0)
  - 0 (自動)
  - 1 (SSE2)
  - 2 (AVX2)

## 更新メモ
#### 0.04 (2022.05.11)
高ビット深度において、中央にフィルタの適用されない領域があったのを修正。

#### 0.03 (2021.02.01)
可能な場合、フレームの情報を保持するように。
( pull request いただいた Asd-g 様、ありがとうございます！)

#### 0.02 (2020.09.05)
16bit時の桁あふれ防止。

#### 0.01 (2020.09.05)  
Avisynth+の高ビット深度に対応。  
AVX2に対応。  
桁あふれによって正常に処理できないことがあったのを修正。

#### 0.00 (2012.03.06)  
初期版
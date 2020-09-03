﻿
# エッジレベル調整 (Avisynth版)

[がらくたハウスのがらくた置き場様](http://www.geocities.jp/flash3kyuu/)の
エッジレベル調整(edgelevelMT.auf)をAvisynthに移植したものです。

また、AVX2 / SSE2 で高速化されています。
環境に合わせて、最速のものが自動的に選択されます。


## ダウンロード & 更新履歴
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-11.html)

## 想定動作環境

Win 10 (x64)  
Avisynth+ (x86/x64)

## 注意事項

無保証です。自己責任で使用してください。  
このフィルタを使用したことによる、いかなる損害・トラブルについても責任を負いません。


## オプション

### edgelevel(clip, int "strength", int "threshold", int "bc", int "wc", int "thread", int "asm")

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

- asm (0 ～ 2, デフォルト 0)
  - 0 (自動)
  - 1 (SSE2)
  - 2 (AVX2)

## 更新メモ
#### 2020.09.04
Avisynth+の高ビット深度に対応。
AVX2に対応。
桁あふれによって正常に処理できないことがあったのを修正。

#### 2012.03.06  
初期版
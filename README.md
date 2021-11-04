# TPS-Oct2021
10月に行われたkaggleのテーブルデータコンペ

## Basic
### Description
Kaggleコンペティションは非常に楽しく、やりがいのあるものですが、データサイエンスを始めて間もない人にとっては敷居が高いものでもあります。これまでKaggleは、注目のコンペティションよりも親しみやすく、初心者にもやさしいPlaygroundコンペティションを数多く開催してきました。

2021年には、コミュニティのためにこれらのコンペティションをより一貫して提供するために、新しい試みを行います。毎月1日に1ヶ月間の表形式のプレイグラウンド大会を開始し、十分な関心と参加がある限り、この実験を続けます。

このコンペティションの目的は、楽しく、誰にでも親しみやすい表形式のデータセットを提供することです。これらのコンペティションは、「タイタニック入門」と「注目のコンペティション」の中間的なものを探している人に最適です。もしあなたが既に大会のマスターやグランドマスターであれば、これらの大会はそれほど難しいものではないでしょう。リーダーボードが飽和状態にならないようにすることをお勧めします。

各月のコンペティションでは、上位3チームにKaggleグッズを提供します。また、このコンペは学習を目的としたものであるため、チームサイズは3人までとしています。

このコンペティションで使用されるデータセットは、合成データですが、実際のデータセットに基づいており、CTGANを使って生成されています。元のデータセットは、様々な化学的特性を持つ分子の生物学的反応を予測するものです。特徴は匿名化されていますが、実世界の特徴に関連する特性を持っています。

頑張って、楽しんでください。

スコアアップのためのアイデアは、Kaggle LearnのIntro to Machine LearningとIntermediate Machine Learningのコースをご覧ください。

www.DeepL.com/Translator（無料版）で翻訳しました。

### Feature
|  f1~f241  |  f241~  |
| ---- | ---- |
|  numericdata  |  categoricaldata  |


## Log
### 20211029
- start!
#### 現状
- アンサンブル学習のために以下のモデルを作成完了
- cat

|  B_f1  |  B_f2  | B_f3 | f_c1 | f_c2 |
| ---- | ---- | ---- | ---- | ---- |
|  CV:0.6959304214  |  0.8546571732  | 0.6987179518 | 0.8549021482   | 0.854847163 |

- lgbm  

|  B_f1  |  B_f2  | B_f3 | f_c1 | f_c2 |
| ---- | ---- | ---- | ---- | ---- |
|  CV:0.697476  |  0.855393  | 0.700225 | 0.855614 | 0.855219  |

- xgb

|  B_f1  |  B_f2  | B_f3 | f_c1 | f_c2 |
| ---- | ---- | ---- | ---- | ---- |
|  CV:0.69508 |  0.85442  | 0.69777 | 0.85442 | 0.85444 |

- NN(keras)

|  NN1  |  NN2  | NN3 | 
| ---- | ---- | ---- |
|  CV:0.8391168353110304  | 0.8375493983782291   | 0.8377000380659657 | 

- モデル詳細

|B_f1|B_f2|B_f3|  NN1  |  NN2  | NN3 | 
| ---- | ---- | ---- | ---- | ---- | ---- |
| f22, f43, f242 ~ f284を削除  | f242 ~ f284を削除 | f22,f43を削除  | L2正規化+ドロップアウト | NN1+B_f2を特徴量から削除  | NN2+B_f2をクラスタリングした特徴量を5つ追加 |

| f_c1 | f_c2|
| ---- | ---- |
| B_f2を削除し、B_f2をクラスタリング下特徴量を3つ追加 | 特徴量重要度50未満のカラムを削除、特徴量重要度上位30個からクラスタリングした特徴量3つ追加 |


#### To do
- [ ] アンサンブル学習2ステージ目で何個モデルをつくるか
- [ ] 3ステージ目で最終提出するものを作成
- [ ] もしかしたら、1ステージ目のモデルを新たに作るかも

![スタッキング](https://user-images.githubusercontent.com/93358183/139364271-bf33ab3f-c24f-45be-bbe6-a452600d5bdc.png)

- 現所モデルは21個ある(1ステージ）
- そのため、21個のうち7個ランダムに選択し、3つのスタッキングモデルを生成する！ (2ステージ）
- その3つから1つのモデルを作る！（3ステージ）

#### 結果

|model | CV | LB |  
| ---- | ---- | ---- |  
| 'cat3_fc1', 'lgbm3_fc1', 'cat1', 'lgbm3_fc2', 'cat2_B2', 'lgbm1', 'xgb2_B2' | 0.8573926959907876 | 0.85642 |
| 'cat2_B3', 'xgb3_fc1', 'xgb2_B1', 'lgbm2_B2', 'cat2_B1', 'xgb3_fc2', 'NN3' | 0.8566027198974991 | 0.85356 |
| 'lgbm2_B1', 'NN2', 'xgb1', 'NN1', 'cat3_fc2', 'lgbm2_B3', 'xgb2_B3' | 0.8568797837281886 | 0.85598 |

|model | CV | LB |  
| ---- | ---- | ---- |  
| 'NN1', 'cat2_B2', 'xgb3_fc2', 'xgb2_B3', 'xgb3_fc1', 'lgbm1', 'lgbm3_fc2 | 0.8574521048266853 |  |
| 'NN2', 'lgbm2_B2', 'cat2_B3', 'cat3_fc2', 'lgbm2_B3', 'xgb1', 'xgb2_B2' | 0.8548178740981378 |  |
| 'lgbm2_B1', 'NN2', 'xgb1', 'NN1', 'cat3_fc2', 'lgbm2_B3', 'xgb2_B3' | 0.8569915715933696 |  |

|model | CV | LB |  
| ---- | ---- | ---- |  
| 'lgbm1', 'lgbm3_fc1', 'xgb2_B1', 'xgb2_B2', 'xgb1', 'lgbm2_B1', 'cat3_fc1' | 0.8574467951016895 |  |
| 'xgb2_B3', 'NN3', 'NN1', 'cat3_fc2', 'lgbm3_fc2', 'cat2_B3', 'cat1' | 0.856891739809438 |  |
| 'xgb3_fc2', 'xgb3_fc1', 'cat2_B1', 'NN2', 'cat2_B2', 'lgbm2_B3', 'lgbm2_B2' | 0.8567193199172081 |  |

|model | CV | LB |  
| ---- | ---- | ---- |  
| 'cat2_B3', 'NN3', 'NN2', 'lgbm2_B1', 'lgbm2_B3', 'xgb3_fc2', 'lgbm1' | 0.8574672850509685 |  |
| 'cat3_fc1', 'xgb1', 'xgb2_B3', 'xgb2_B2', 'lgbm3_fc2', 'NN1', 'cat3_fc2' | 0.8568912736789993 |  |
| 'cat2_B2', 'cat1', 'cat2_B1', 'lgbm3_fc1', 'xgb2_B1', 'xgb3_fc1', 'lgbm2_B2' | 0.8569983328997315 |  |

|model | CV | LB |  
| ---- | ---- | ---- |  
| 'NN2', 'xgb2_B2', 'xgb2_B3', 'xgb2_B1', 'NN3', 'cat2_B2', 'cat2_B3' | 0.8562205149678825 |  |
| 'lgbm2_B1', 'lgbm1', 'xgb3_fc1', 'xgb3_fc2', 'lgbm2_B3', 'lgbm3_fc2', 'lgbm3_fc1' | 0.8574390667044178 |  |
| 'cat2_B1', 'cat3_fc1', 'lgbm2_B2', 'xgb1', 'cat3_fc2', 'NN1', 'cat1' | 0.8571404029334051 |  |

- Codeを高スコア順に見ると複数データを加重平均したものが高いスコアを得ているみたい...

 https://www.kaggle.com/vamsikrishnab/exploring-submissions-and-power-averaging/notebook
 
- 自分のスタッキングしたものと上urlのデータを平均化するといいデータが得られるかも？


### 20211030

#### **Power Average**

**使用したモデル**

|model | CV | LB |  
| ---- | ---- | ---- |  
| 'NN3', 'cat1', 'NN1', 'xgb2_B3', 'cat2_B3', 'cat2_B1', 'lgbm1' | 0.8574906266129305 | 0.85650 |
| 'cat2_B3', 'NN3', 'NN2', 'lgbm2_B1', 'lgbm2_B3', 'xgb3_fc2', 'lgbm1' | 0.8574672850509685 | 0.85644 |
| clx1, clx2_B1, cls2_B2, clx2_B3, clx3_fc1, clx_fc2 | 0.8574618186258252 | 0.85646 |

clx...cat, lgbm, xgbの略


**結果**

上記の3つのモデルを使ってパワー平均化を行った。

|Public Score|
| ---------- |
|   0.85649  |

- かなり高い数値になったが、使用したモデルを超えてないので微妙？
- PrivateScoreで抜く可能性あり！


### 20211031(**Final submission deadline**)

- JOE COOPERさんの手法を使ったりしたが過学習が起こり残念な結果に...

https://www.kaggle.com/joecooper/tps-oct-joes-sandpit?scriptVersionId=78350199

|CV|Public Score|
|---| ---------- |
| 0.8570173280476039 |   0.82604  |


### 20211101（**Close**)

- 結局、PublicScoreが高かったパワー平均化とそれに使用したモデルをPrivateに提出した。

#### 結果

|model | CV | P | Private |
| ---- | ---- | ---- | ---- |
| 'NN3', 'cat1', 'NN1', 'xgb2_B3', 'cat2_B3', 'cat2_B1', 'lgbm1' | 0.8574906266129305 | 0.85650 | 0.85641 |
| PowerAverage |  | 0.85649 | 0.85639 |

- LBの順位は141位(全体の13%)だった。
- 悪くはない順位だったが、少し物足りない結果に...



### 20211103（After closing)

- クラスタリングのエルボー法をやってみた　
- コード->https://github.com/fa545506/TPS-Oct2021/blob/main/nb/n01-fclutreing-elbow.ipynb
  - B_featureの正規化なし
    
    肘が見当たらないw
    
    恐らく6あたり？
    
    ![image](https://user-images.githubusercontent.com/93358183/140011994-0ed8f801-7c80-4875-8fba-9496fd85fd52.png)

  -  B_featureの正規化あり（標準化）
     
     うーん、意味なしw
     
  　　![image](https://user-images.githubusercontent.com/93358183/140011980-c6f0badf-532a-4a55-88d8-6fb247128529.png)

  - B_feature以外
  
    ![image](https://user-images.githubusercontent.com/93358183/140011961-80314b98-f23d-4c5b-a3fd-f4591a7f14c1.png)

  - B_featureのみ
   - 正規化
    
     ![image](https://user-images.githubusercontent.com/93358183/140011873-10c5ecc3-346b-424e-b804-f5a62a969544.png)
  
   - 正規化なし
     
     ![image](https://user-images.githubusercontent.com/93358183/140011900-047101e9-5df9-47c7-9735-884f81db3418.png)


### 20211104

- Optunaでパラメータチューニングやってみた
- ほとんどオリジナルと変化なかったが、PublicスコアとCVが若干変化した。
- 全パラのほうは4h実行、部パラのほうは100回試行
- もっと時間を掛けたら大きく変化するかも？
　- 初期パラメータ
   
　  ####結果
    
    |model | CV | P | Private |
    | ---- | ---- | ---- | ---- |
    | 'NN3', 'cat1', 'NN1', 'xgb2_B3', 'cat2_B3', 'cat2_B1', 'lgbm1' | 0.8574905274728372 | 0.85650 | 0.85641 |
 
  - 全部のパラメータチューニング　->
   　
    ####結果
    
    |model | CV | P | Private |
    | ---- | ---- | ---- | ---- |
    | 'NN3', 'cat1', 'NN1', 'xgb2_B3', 'cat2_B3', 'cat2_B1', 'lgbm1' | 0.8574726465060132 | 0.85652 | 0.85641 |
  
  - 一部のパラメータチューニング　->

    ####結果
    
    |model | CV | P | Private |
    | ---- | ---- | ---- | ---- |
    | 'NN3', 'cat1', 'NN1', 'xgb2_B3', 'cat2_B3', 'cat2_B1', 'lgbm1' | 0.8574927843549608 | 0.85651 | 0.85641 |
  
  
 

# 振り返り

## 今回得たこと

- 大量のデータに対する処理の仕方
  - pandasで読み込むのではなく、DataTableで読み込む
 
    https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/276162
  
  - reduce_memory_usage関数でデータのメモリ削減
  
    https://github.com/fa545506/TPS-Oct2021/blob/main/nb/reduce_memory_usage.py
  
- アンサンブル学習を活用することができた
  - 今回使用したのはスタッキング、パワー平均化の2つの手法
  - アンサンブル学習によって大きくスコア伸ばすことができた。
  
 
- 特徴量クラスタリング
  - 今回、大量のバイナリデータが含まれていたので、その取り扱いとしてクラスタリング(k-means)を取り入れた
  
    https://yaakublog.com/kmeans_clustering
  
  

## 今後の課題

- グラフの活用
  - モデル間の相関
  - 特徴量間の相関->独立性はあるか、
  - 特徴量と正解データの相関

- 統計知識の不足
  - アンサンブル学習のときのバイアス、バリアンスやピアソン値、スミルノフ統計値を考慮したモデルの組み合わせ

　　https://www.codexa.net/what-is-ensemble-learning/
  
　　https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/51058#290767
  
- モデルのパラメータチューニング
  - 今回全くしなかった
  - Optunaを使って最適化するともっとスコアが向上していたかも

- kaggle日記
  - ノートブックの整理、番号付け->ex) nb001_Lgbm, nb002_NN, .... 
  - トレーサビリティ->gitにまとめたスコア結果等とノートブックが関連されておらず後から探すのが大変だった。
  - 計画を立てる->実行の順番でやることで、状況が整理される。
  - 思い付きでがつがつやると何がなんだがよくわからなくなる。
  - コンペ締め切り3日前から慌ててやりだしたが、やる前と比べ状況が整理できた。
  - 次回は最初から日記を書いていきたい
  
  　https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068
 
 

## その他の学び
- k-meansの最適なクラスタ数
  - エルボー法とシルエット分析
  
    https://qiita.com/shuva/items/bcf700bd32ae2bbb63c7#elbow-method
    
    https://www.kaggle.com/motchan/tps-oct-2021-kmeans/comments
 
 - Bouta-SHAPを用いた特徴量選択
   - SHAPはどの特徴量がスコアに寄与したかを教えてくれるもの
   - 今回の訓練データは特徴量が非常に多かったのでいかに有効な特徴量を選べるかがカギになった。

    https://www.kaggle.com/lucamassaron/feature-selection-using-boruta-shap
    
- seed値を変えたモデルを複数作成し、それを平均化する
  
  https://www.kaggle.com/adamwurdits/15-lgbms-trained-on-20-seeds-dataset-included/notebook


- 様々なモデル
  -  ElasticNet
  -  Linear Discriminant Analysis->スタッキングのレベル2で使用されることが多い
  -  RidgeCV


- 今回のコンペでNN系のモデルが軒並みCV、Public、Private全てで低かった。
  ->その原因は特徴量の数が多く、ノイズが発生したためスコアが伸びなかった。
  ->その対策:いかに最適な特徴量を得られるかがカギ、手法としてVariable Selection Networks
            https://arxiv.org/abs/1912.09363
            https://keras.io/examples/structured_data/classification_with_grn_and_vsn/
            https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284511

- Leave One Feature Out (LOFO)による特徴量重要度の算出
  - lgbmやNNなどのモデルに問わず行える
  - モデルごとに異なる重要度を使う必要がなくなる
 
    https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/281993
  　https://www.kaggle.com/c/microsoft-malware-prediction/discussion/79415

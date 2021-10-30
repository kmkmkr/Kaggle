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

|model | CV | PB |  
| ---- | ---- | ---- |  
| 'cat3_fc1', 'lgbm3_fc1', 'cat1', 'lgbm3_fc2', 'cat2_B2', 'lgbm1', 'xgb2_B2' | 0.8573926959907876 | 0.85642 |
| 'cat2_B3', 'xgb3_fc1', 'xgb2_B1', 'lgbm2_B2', 'cat2_B1', 'xgb3_fc2', 'NN3' | 0.8566027198974991 | 0.85356 |
| 'lgbm2_B1', 'NN2', 'xgb1', 'NN1', 'cat3_fc2', 'lgbm2_B3', 'xgb2_B3' | 0.8568797837281886 | 0.85598 |

|model | CV | PB |  
| ---- | ---- | ---- |  
| 'NN1', 'cat2_B2', 'xgb3_fc2', 'xgb2_B3', 'xgb3_fc1', 'lgbm1', 'lgbm3_fc2 | 0.8574521048266853 |  |
| 'NN2', 'lgbm2_B2', 'cat2_B3', 'cat3_fc2', 'lgbm2_B3', 'xgb1', 'xgb2_B2' | 0.8548178740981378 |  |
| 'lgbm2_B1', 'NN2', 'xgb1', 'NN1', 'cat3_fc2', 'lgbm2_B3', 'xgb2_B3' | 0.8569915715933696 |  |

|model | CV | PB |  
| ---- | ---- | ---- |  
| 'lgbm1', 'lgbm3_fc1', 'xgb2_B1', 'xgb2_B2', 'xgb1', 'lgbm2_B1', 'cat3_fc1'2 | 0.8574467951016895 |  |
| 'xgb2_B3', 'NN3', 'NN1', 'cat3_fc2', 'lgbm3_fc2', 'cat2_B3', 'cat1' | 0.856891739809438 |  |
| 'xgb3_fc2', 'xgb3_fc1', 'cat2_B1', 'NN2', 'cat2_B2', 'lgbm2_B3', 'lgbm2_B2' | 0.8567193199172081 |  |

- Codeを高スコア順に見ると複数データを加重平均したものが高いスコアを得ているみたい...

 https://www.kaggle.com/vamsikrishnab/exploring-submissions-and-power-averaging/notebook
 
- 自分のスタッキングしたものと上urlのデータを平均化するといいデータが得られるかも？


### 20211030



### 20211031(**Final submission deadline**)





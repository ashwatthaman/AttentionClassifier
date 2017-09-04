AttentionClassifier（https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf）のchainer実装です。
テキストの分類をする時に、どの箇所が分類結果に効いているのかを分析する事ができます。

上記論文では文単位の処理からドキュメント単位の処理を階層的（Hierarchical）に行うモデルですが、
chainer2.02で動作確認。

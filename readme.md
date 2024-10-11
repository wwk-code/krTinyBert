# SentimentAnalysis nluBenchmark分支

### 1.分支说明：
    本分支为 SentimentAnalysis项目的测试分支，用于测试应用中的 mobileBERT 模型分别在骁龙8Gen3上的NPU以及CPU上推理时性能对比、资源消耗对比,具体的测试记录请参考 benchmark/record.md 文件内容

### 2.应用说明:
    本应用为文本情感分析软件，应用中集成了训练好的MobileBERT模型，可接受英文文本输入，然后执行推理，输出这个文本中的积极情感态度以及消极情感态度的占比 。使用方法如下: 用户在文本框中输入英文文本，随后点击: "PREDICT DSP" 或 "PREDICT CPU",随后便可在下方文本输出框中看到模型执行推理的结果以及测试日志的输出。

### 3.可用 apk:
    可直接使用项目目录下的 app-release.apk (可能由于软件太大，使用了 git lfs 管理，需要用户拉取项目后使用 git lfs pull拉取)

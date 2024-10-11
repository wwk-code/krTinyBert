# krTinybert 主分支

### 1.分支说明：
    本分支为 krTinybert 主分支，目前此分支已经可以跑通fp32精度的tinyBert在CPU和DSP下的推理，推理结果与原始onnx推理结果完全匹配 

### 2.应用说明:
    本应用为文本情感分析软件，应用中集成了经过sst2数据集微调好的tinyBert模型，可接受英文文本输入，然后执行推理，输出这个文本中的积极情感态度以及消极情感态度的占比，最终这些数据可以转化为情感预测标签。
 app使用方法如下: 按下Benchmark键，app会生成模型输出的logits文件，具体内容请看MainActivity.java中的oncreate方法


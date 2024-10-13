# 情感分析APP - asr 分支

目前本分支已实现实时语音识别功能，但采用的推理框架为: Sherpa-onnx,且调用的是晓龙CPU芯片的进行推理，而非NPU

### 1.APP效果:
    目前本分支部署的是  main/assets/sherpa-onnx-streaming-paraformer-bilingual-zh-en 下的模型，此模型参数量为230MB左右，其是在线模型，为实时模型，实时语音识别能力较强.
    进入软件界面后,按下start键即可开始实时的语音识别并显示到文本界面.按下 stop 键即可清空文本界面,再次按下start则再次重复.

### 2.基于的模型:
    说明: 此分支基于的模型是阿里达摩院开源的基于Paraformer架构的中英文语音识别模型，其为实时语音识别模型
    仓库链接:  https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english
    进入连接后，选择:  csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en (Chinese + English) 条目的模型，下载模型后将整个文件夹移动至项目的 src/main/assets 目录下

### 3. jniLibs文件:
    参考 sherpa-onnx github 仓库的 build-android-arm64-v8a.sh 脚本，可构建出相应的库文件，本项目中所依赖的相关库文件为: arm64-v8a/libonnxruntime.so、libsherpa-onnx-jni.so
    sherpa-onnx github仓库链接: https://github.com/k2-fsa/sherpa-onnx/tree/master


### 4. 项目使用教程:
    可直接使用。若要构建运行完整的项目，可能需要使用  git lfs pull 拉取被lfs追踪的文件(Android Studio可能会自动使用 lfs 拉取，若没自动拉取需要用户手动拉取),具体被追踪的文件可查看 .gitattributes 文件内容 

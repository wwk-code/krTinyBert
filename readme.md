# 离线实时语音识别安卓APP - onlineAsrShepaOnnx分支

目前本分支已实现实时语音识别功能，但采用的推理框架为: Sherpa-onnx,且调用的是晓龙Gen3芯片的CPU进行推理，而非NPU

### 0. 输出Apk:
    本分支项目构建后输出的apk为项目目录下的 asrAgent.apk 文件

### 1.APP效果:
    目前本分支部署的是  main/assets/sherpa-onnx-streaming-paraformer-bilingual-zh-en 下的模型，此模型参数量为230MB左右，其是在线模型，为实时模型，实时语音识别能力较强

### 2.基于的模型:
    说明: 此分支基于的模型是阿里达摩院开源的基于Paraformer架构的中英文语音识别模型，其为实时语音识别模型
    仓库链接:  https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english
    进入连接后，选择:  csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en (Chinese + English) 条目的模型，下载模型后将整个文件夹移动至项目的 src/main/assets 目录下

### 3. jniLibs文件:
    参考 sherpa-onnx github 仓库的 build-android-arm64-v8a.sh 脚本，可构建出相应的库文件，本项目中所依赖的相关库文件为: arm64-v8a/libonnxruntime.so、libsherpa-onnx-jni.so
    sherpa-onnx github仓库链接: https://github.com/k2-fsa/sherpa-onnx/tree/master


### 4. 项目使用教程:
    本分支下已有 app-debug.apk ，可直接使用。若要构建运行完整的项目，可能需要使用  git lfs pull 拉取被lfs追踪的文件(Android Studio可能会自动使用 lfs 拉取，若没自动拉取需要用户手动拉取),具体被追踪的文件可查看 .gitattributes 文件内容 

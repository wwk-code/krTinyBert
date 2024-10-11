#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <sstream>
#include "hpp/inference.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_qualcomm_qti_sentimentanalysis_MainActivity_queryRuntimes(
        JNIEnv* env,
        jobject /* this */,
        jstring native_dir_path) {
    const char *cstr = env->GetStringUTFChars(native_dir_path, NULL);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string runT_Status;
    std::string nativeLibPath = std::string(cstr);

//    runT_Status += "\nLibs Path : " + nativeLibPath + "\n";

    if (!SetAdspLibraryPath(nativeLibPath)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "Failed to set ADSP Library Path\n");
        runT_Status += "\nFailed to set ADSP Library Path\nTerminating";
        return env->NewStringUTF(runT_Status.c_str());
    }
    // ====================================================================================== //
    runT_Status = "Querying Runtimes : \n\n";
    // DSP unsignedPD check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP,zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "UnsignedPD DSP runtime : Absent\n");
        runT_Status += "UnsignedPD DSP runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "UnsignedPD DSP runtime : Present\n");
        runT_Status += "UnsignedPD DSP runtime : Present\n";
    }
    // DSP signedPD check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "DSP runtime : Absent\n");
        runT_Status += "DSP runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "DSP runtime : Present\n");
        runT_Status += "DSP runtime : Present\n";
    }
    // GPU check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "GPU runtime : Absent\n");
        runT_Status += "GPU runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "GPU runtime : Present\n");
        runT_Status += "GPU runtime : Present\n";
    }
    // CPU check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "CPU runtime : Absent\n");
        runT_Status += "CPU runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "CPU runtime : Present\n");
        runT_Status += "CPU runtime : Present\n";
    }

    return env->NewStringUTF(runT_Status.c_str());
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_qualcomm_qti_sentimentanalysis_MainActivity_initSNPE(JNIEnv *env, jobject thiz,
                                                              jobject asset_manager) {
    LOGI("Reading SNPE DLC ...");
    std::string result;
//    const char *fileName = "model_int8.dlc";
    const char *fileName = "tinyBert.dlc";
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset = AAssetManager_open(mgr, fileName, AASSET_MODE_UNKNOWN);
    if (NULL == asset) {
        LOGE("Failed to load ASSET, needed to load DLC\n");
        result = "Failed to load ASSET, needed to load DLC\n";
        return env->NewStringUTF(result.c_str());
    }
    long dlc_size = AAsset_getLength(asset);
    LOGI("DLC Size = %ld MB\n", dlc_size / (1024*1024));
//    result += "DLC Size = " + std::to_string(dlc_size);
    char* dlc_buffer = (char*) malloc(sizeof(char) * dlc_size);
    AAsset_read(asset, dlc_buffer, dlc_size);

    result += "\n\nBuilding Model DLC Network:\n";
    result += build_network(reinterpret_cast<const uint8_t *>(dlc_buffer), dlc_size);

    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_qualcomm_qti_sentimentanalysis_MainActivity_inferSNPE(JNIEnv *env, jobject thiz,
                                                               jstring runtime, jintArray input_ids,
                                                               jintArray attention_mask,
                                                               jintArray arrayLengths,
                                                               jint executeTimes) {
    std::string return_msg;
    jint * inp_id_array;
    jint * mask_array;
    jint * arr_len_array;

    const char *accl_name = env->GetStringUTFChars(runtime, NULL);
    env->ReleaseStringUTFChars(runtime, accl_name);
    std::string backend = std::string(accl_name);

    // get a pointer to the array
    inp_id_array = env->GetIntArrayElements(input_ids, NULL);
    mask_array = env->GetIntArrayElements(attention_mask, NULL);
    arr_len_array = env->GetIntArrayElements(arrayLengths, NULL);
    // 模型输出 output1
    int output1_len = 2;
    std::vector<float> output1(output1_len);
    jfloatArray output1_retArray = env->NewFloatArray(output1_len);

    // do some exception checking
    if (inp_id_array == NULL ||  mask_array == NULL || arr_len_array == NULL) {
        env->SetFloatArrayRegion(output1_retArray,0,output1_len,output1.data());
        return output1_retArray;
    }
    std::vector<int*> inputVec { inp_id_array, mask_array};
    std::unordered_map<std::string ,int*> inputDataMap = {
            {"input_ids",inp_id_array},
            {"attention_mask",mask_array},
    };
    // Execute the input buffer map on the model with SNPE
    auto tic = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    for (int i = executeTimes;i > 0;i--) return_msg = execute_net(inputDataMap, arr_len_array, backend,output1);

    auto toc = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    LOG_BENCHMARK("Average Native inference time for %s = %f ms", accl_name, static_cast<double>((toc-tic)*1.0) / executeTimes);


    LOGI("SNPE JNI : %s", return_msg.c_str());

    env->SetFloatArrayRegion(output1_retArray,0,output1_len,output1.data());

    // ===================================================================== //
    // release the memory so java can have it again
    env->ReleaseIntArrayElements(input_ids, inp_id_array, 0);
    env->ReleaseIntArrayElements(attention_mask, mask_array, 0);
    env->ReleaseIntArrayElements(arrayLengths, arr_len_array, 0);

    return output1_retArray;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_qualcomm_qti_sentimentanalysis_MainActivity_exportDSPENV(JNIEnv *env, jobject thiz,jstring nativeLibPath) {
    // Convert jstring to const char*
    const char* path_chars = env->GetStringUTFChars(nativeLibPath, nullptr);
    std::string nativeLibPath_str(path_chars);
    std::stringstream path;
    path << nativeLibPath_str << ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    return setenv("ADSP_LIBRARY_PATH", path.str().c_str(), 1 /*override*/) == 0;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_qualcomm_qti_sentimentanalysis_MainActivity_getEnv(JNIEnv *env, jobject thiz, jstring var_name) {
    const char* var_name_chars = env->GetStringUTFChars(var_name, nullptr);
    const char* var_value = getenv(var_name_chars);
    env->ReleaseStringUTFChars(var_name, var_name_chars);
    return env->NewStringUTF(var_value ? var_value : "");
}












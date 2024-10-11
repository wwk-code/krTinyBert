#include <jni.h>
#include <string>
#include <iostream>

#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <hpp/inference.h>

#include "android/log.h"

#include "hpp/CheckRuntime.hpp"
#include "hpp/LoadContainer.hpp"
#include "hpp/SetBuilderOptions.hpp"
#include "hpp/LoadInputTensor.hpp"
#include "hpp/CreateUserBuffer.hpp"
#include "hpp/Util.hpp"

std::unique_ptr<zdl::SNPE::SNPE> snpe_dsp;
std::unique_ptr<zdl::SNPE::SNPE> snpe_cpu;
static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
static zdl::DlSystem::RuntimeList runtimeList;
bool useUserSuppliedBuffers = true;
bool useIntBuffer = false;

std::string dlc_path;

std::string build_network(const uint8_t * dlc_buffer, const size_t dlc_size)
{
    std::string outputLogger;
    bool usingInitCaching = true;

    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = loadContainerFromBuffer(dlc_buffer, dlc_size);
    if (container == nullptr) {
        LOGE("Error while opening the container file.");
        return "Error while opening the container file.\n";
    }

    runtimeList.clear();
    // Build for DSP runtime
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
    if(runtime != zdl::DlSystem::Runtime_t::UNSET)
    {
        bool ret = runtimeList.add(checkRuntime(runtime));
            if(ret == false){
                LOGE("Cannot set runtime");
                return outputLogger + "\nCannot set runtime";
            }
    } else return outputLogger + "\nCannot set runtime";

    snpe_dsp = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);

    if (snpe_dsp == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for DSP runtime");
        outputLogger += "Model Prepare failed for DSP runtime";
//        return outputLogger + "SNPE Prepare failed for DSP runtime";
    }
    if (usingInitCaching) {
        if (container->save(dlc_path)) {
            LOGI("Saved container into archive successfully");
            outputLogger += "\nSaved container cache";
        } else LOGE("Failed to save container into archive");
    }

    // Build for CPU runtime
    runtimeList.clear();
    runtime = zdl::DlSystem::Runtime_t::CPU;
    snpe_cpu = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);
    if (snpe_cpu == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for CPU runtime");
        return outputLogger + "Model Prepare failed for CPU runtime";
    }

    outputLogger += "\Model Network Prepare success !!!\n";
    return outputLogger;
}

static zdl::DlSystem::TensorMap outputTensorMap;

// input vector, runtime
std::string execute_net(std::unordered_map<std::string ,int*> inputDataMap, int arrayLengths[], std::string runtime, std::vector<float> &output1) {
    bool execStatus;
    std::string sentiment;
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
//    LOGI("Runtime received = %s", runtime.c_str());

    // Transfer object properties
    if (runtime == "CPU") {
        snpe = std::move(snpe_cpu);
        LOGI("Executing on CPU runtime...");
    } else snpe = std::move(snpe_dsp);

    zdl::DlSystem::UserBufferMap inputMap, outputMap;
    std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
    std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;

    // create UB_TF_N type buffer, if : useIntBuffer=True
    int bitWidth = 32;

    LOGI("Using UserBuffer with bit-width = %d", bitWidth);

    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, useIntBuffer, bitWidth);

    std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, useIntBuffer, bitWidth);
    // 输入数据类型均为 int32
    bitWidth = 32;

    if(!loadInputUserBuffer(applicationInputBuffers, snpe, inputDataMap, arrayLengths, inputMap, bitWidth))
        return "\nFailed to load Input UserBuffer";



    // ///////////////////////////////////////////////////////////   调 SNPE 库执行模型推理
    execStatus = snpe->execute(inputMap, outputMap);



    // Save the execution results only if successful
    if (execStatus == true) {
        LOGI("SNPE Exec Success !!!");
        // save output tensor
        size_t batchSize = 1;
        bitWidth = 32;
        if(!getOutput(outputMap, applicationOutputBuffers, output1, batchSize, bitWidth))
            return "\nFailed to Save Output Tensor";
    } else return "\DNN Execute Failed\n";

    // Transfer object properties
    if (runtime == "CPU") {
        snpe_cpu = std::move(snpe);
        LOGI("Transferred back object to CPU runtime...");
    } else snpe_dsp = std::move(snpe);

    return sentiment;
}

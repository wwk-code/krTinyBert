#include <jni.h>
#include <string>
#include <iostream>
#include "android/log.h"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/DlSystem/DlVersion.hpp"
#include "zdl/DlSystem/DlEnums.hpp"
#include "zdl/DlSystem/String.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"
#include "zdl/DlSystem/ITensor.hpp"
#include "zdl/DlSystem/StringList.hpp"
#include "zdl/DlSystem/TensorMap.hpp"
#include "zdl/DlSystem/TensorShape.hpp"
#include "DlSystem/ITensorFactory.hpp"

#include "hpp/LoadInputTensor.hpp"
#include "hpp/Util.hpp"
#include "inference.h"


bool SetAdspLibraryPath(std::string nativeLibPath) {
    nativeLibPath += ";/data/local/tmp/mv_dlc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";

    __android_log_print(ANDROID_LOG_INFO, "SNPE ", "ADSP Lib Path = %s \n", nativeLibPath.c_str());
    std::cout << "ADSP Lib Path = " << nativeLibPath << std::endl;

    return setenv("ADSP_LIBRARY_PATH", nativeLibPath.c_str(), 1 /*override*/) == 0;
}


std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(containerPath.c_str()));
    return container;
}

std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromBuffer(const uint8_t * buffer, const size_t size)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(buffer, size);
    return container;
}

zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

    LOGI("SNPE Version = %s", Version.asString().c_str()); //Print Version number

//    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
//        LOGE("Selected runtime not present. Falling back to GPU.");
//        runtime = zdl::DlSystem::Runtime_t::GPU;
//        if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)){
//            LOGE("Selected runtime not present. Falling back to CPU.");
//            runtime = zdl::DlSystem::Runtime_t::CPU;
//        }
//    }

    return runtime;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   bool useCaching)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    std::string platformOptionStr = "useAdaptivePD:ON";
//  if (isSignedStatus == UNSIGNED_PD) {
        // use unsignedPD feature for untrusted app.
        // platformOptionStr += "unsignedPD:ON";
//  }
    zdl::DlSystem::PlatformConfig platformConfig;
    bool setSuccess = platformConfig.setPlatformOptions(platformOptionStr);
    if (!setSuccess)
        LOGE("=========> failed to set platformconfig: %s", platformOptionStr.c_str());
    else
        LOGI("=========> platformconfig set: %s", platformOptionStr.c_str());

    bool isValid = platformConfig.isOptionsValid();
    if (!isValid)
        LOGE("=========> platformconfig option is invalid");
    else
        LOGI("=========> platformconfig option: valid");


    snpe = snpeBuilder.setOutputLayers({})
            .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::BURST)
            .setExecutionPriorityHint(
                    zdl::DlSystem::ExecutionPriorityHint_t::HIGH)
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(useCaching)
            .build();
    return snpe;
}

std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE>& snpe , std::vector<float>& inp_raw) {
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert (strList.size() == 1);

    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;

    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

    if (input->getSize() != inp_raw.size()) {
        std::string errStr = "Size of input does not match network.\n Expecting: " + std::to_string(input->getSize());
        errStr +=  "; Got: " + std::to_string(inp_raw.size()) + "\n";
        LOGE("%s",errStr.c_str());
        return nullptr;
    }

    /* Copy the loaded input file contents into the networks input tensor.
    SNPE's ITensor supports C++ STL functions like std::copy() */
    std::copy(inp_raw.begin(), inp_raw.end(), input->begin());
    return input;
}

// ==============================User Buffer func=================================== //
// ================================================================================= //
size_t resizable_dim;

size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank, size_t elementSize ){
    if (rank == 0) return 0;
    size_t size = elementSize;
    while (rank--) {
        (*dims == 0) ? size *= resizable_dim : size *= *dims;
        dims++;
    }
    return size;
}

void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char * name,     // snpe->getOutputTensorNames();
                      bool isOutput)
{
    // get attributes of buffer by name
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);

    // calculate the size of buffer required by the input tensor
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();

    size_t tempBufferElementSize = (*bufferAttributesOpt)->getElementSize();
//    if(!isOutput) tempBufferElementSize *= 2;
    const size_t bufferElementSize = tempBufferElementSize;

    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = bufferElementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
//        (bufferShape[i] == 0) ? stride *= getResizableDim() : stride *= bufferShape[i];
        stride *= bufferShape[i];
        strides[i-1] = stride;
    }

    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);

    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingOutput;
    zdl::DlSystem::UserBufferEncodingIntN userBufferEncodingInput(32);


    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));

    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

    if(isOutput) snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                                                            bufSize,
                                                                            strides,
                                                                            &userBufferEncodingOutput));
    else  snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                                                     bufSize,
                                                                     strides,
                                                                     &userBufferEncodingInput));

    if (snpeUserBackedBuffers.back() == nullptr)
        throw std::runtime_error(std::string("Error while creating user buffer."));

    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void createOutputBufferMap(zdl::DlSystem::UserBufferMap& outputMap,
                           std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                           std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                           std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                           bool isTfNBuffer,
                           int bitWidth)
{
    // get input tensor names of the network that need to be populated
    LOGI("Creating Output Buffer");
    const auto& outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

    // create SNPE user buffers for each application storage buffer
    for (const char *name : outputNames) {
//        LOGI("Creating output buffer %s", name);
        createUserBuffer(outputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name,true);
    }
}

void createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                          std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                          bool isTfNBuffer,
                          int bitWidth)
{
//    LOGI("Creating Input Buffer");
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char *name : inputNames) {
//        LOGI("Input Buffer = %s", name);

        // 这里不仅仅会创建 UserBuffer,还会在 inputMap中放入 UserBuffer
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name,false);
    }
}

bool FloatToTfN(uint8_t* out,
                unsigned char& stepEquivalentTo0,
                float& quantizedStepSize,
                float* in,
                size_t numElement,
                int bitWidth)
{
    float trueMin = std::numeric_limits <float>::max();
    float trueMax = std::numeric_limits <float>::min();

    for (size_t i = 0; i < numElement; ++i) {
        trueMin = fmin(trueMin, in[i]);
        trueMax = fmax(trueMax, in[i]);
    }

    double encodingMin;
    double encodingMax;
    double stepCloseTo0;
    double trueBitWidthMax = pow(2, bitWidth) -1;

    if (trueMin > 0.0f) {
        stepCloseTo0 = 0.0;
        encodingMin = 0.0;
        encodingMax = trueMax;
    } else if (trueMax < 0.0f) {
        stepCloseTo0 = trueBitWidthMax;
        encodingMin = trueMin;
        encodingMax = 0.0;
    } else {
        double trueStepSize = static_cast <double>(trueMax - trueMin) / trueBitWidthMax;
        stepCloseTo0 = -trueMin / trueStepSize;
        if (stepCloseTo0==round(stepCloseTo0)) {
            // 0.0 is exactly representable
            encodingMin = trueMin;
            encodingMax = trueMax;
        } else {
            stepCloseTo0 = round(stepCloseTo0);
            encodingMin = (0.0 - stepCloseTo0) * trueStepSize;
            encodingMax = (trueBitWidthMax - stepCloseTo0) * trueStepSize;
        }
    }

    const double minEncodingRange = 0.01;
    double encodingRange = encodingMax - encodingMin;
    quantizedStepSize = encodingRange / trueBitWidthMax;
    stepEquivalentTo0 = static_cast <unsigned char> (round(stepCloseTo0));

    if (encodingRange < minEncodingRange) {
        LOGE("Expect the encoding range to be larger than %f", minEncodingRange);
        LOGE("Got: %f", encodingRange);
        return false;
    } else {
        for (size_t i = 0; i < numElement; ++i) {
            int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);

            if (quantizedValue < 0)
                quantizedValue = 0;
            else if (quantizedValue > (int)trueBitWidthMax)
                quantizedValue = (int)trueBitWidthMax;

            if(bitWidth == 8){
                out[i] = static_cast <uint8_t> (quantizedValue);
            }
            else if(bitWidth == 16){
                uint16_t *temp = (uint16_t *)out;
                temp[i] = static_cast <uint16_t> (quantizedValue);
            }
        }
    }
    return true;
}


bool loadInputUserBuffer(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::unordered_map<std::string ,int*> inputDataMap,
                         int arrayLengths[],
                         zdl::DlSystem::UserBufferMap& inputMap,
                         int bitWidth)
{
    // get input tensor names of the network that need to be populated
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList &inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) LOGI("Processing DNN Input: ");

    if (inputNames.size() != inputDataMap.size())
    {
        throw std::runtime_error("The number of input tensors and input vectors do not match.");
    }

    for (size_t j = 0; j < inputNames.size(); j++)
    {
        const char *name = inputNames.at(j);
        std::string name_str(name);
        // load data on to applicationBuffers
        int* inputVectorData = inputDataMap[name_str];
        int inputVectorLength = arrayLengths[j];
        size_t bufferSize = inputVectorLength * bitWidth / 8;

        std::vector<uint8_t>& userBackedBufferData = applicationBuffers[name];

        // Check if the userBackedBufferData size is the same as bufferSize
        if (userBackedBufferData.size() != bufferSize)
        {
            throw std::runtime_error("The size of user-backed buffer data does not match the input vector size.");
        }

        // Copy inputVectorData to userBackedBufferData
        std::memcpy(userBackedBufferData.data(), inputVectorData, bufferSize);
    }
    return true;
}


bool getOutput(zdl::DlSystem::UserBufferMap& outputMap,
                std::unordered_map<std::string, std::vector<uint8_t>>& applicationOutputBuffers,
                std::vector<float>& output,
                size_t batchSize,
                int bitWidth)
{
    const zdl::DlSystem::StringList &outputBufferNames = outputMap.getUserBufferNames();
    int elementSize = bitWidth / 8;
    for (auto &name : outputBufferNames)
    {
        for (size_t i = 0; i < batchSize; i++)
        {
            auto bufferPtr = outputMap.getUserBuffer(name);
            std::vector<uint8_t>& outputBufferData = applicationOutputBuffers.at(name);
            size_t numElements = outputBufferData.size() / elementSize;
            float* floatArray = reinterpret_cast<float*>(outputBufferData.data());
            output.assign(floatArray, floatArray + numElements);
        }
    }
    return true;
}

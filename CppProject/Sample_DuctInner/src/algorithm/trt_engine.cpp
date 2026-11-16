#include "trt_engine.h"

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

int onnx2engine(std::string onnx_filename, std::string enginefilePath, int batchsize, int imagesize, int type)
{
    // create builder
    TRTLogger logger;
    IBuilder *builder = createInferBuilder(logger);

    // create network
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // create onnxparser
    auto parser = nvonnxparser::createParser(*network, logger);

    // parse model from file
    parser->parseFromFile(onnx_filename.c_str(), 2);
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "[TRTEngine] tensorRT load onnx model sucessful! \n";
    // builder->setMaxBatchSize(batchsize);

    IBuilderConfig *config = builder->createBuilderConfig();
    // config->setMaxWorkspaceSize(1 * (1 << 30));
    //  config->setFlag(BuilderFlag::kGPU_FALLBACK);
    //	config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (type == 1)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (type == 2)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    // printf("%s %d\n", __FUNCTION__, __LINE__);
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(batchsize, 3, imagesize, imagesize));
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(batchsize, 3, imagesize, imagesize));
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(batchsize, 3, imagesize, imagesize));
    config->addOptimizationProfile(profile);

    ICudaEngine *myengine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "[TRTEngine] try to save engine file now" << std::endl;
    std::ofstream p(enginefilePath, std::ios::binary);
    if (!p)
    {
        std::cerr << "[TRTEngine](ERROR) could not open plan output file" << std::endl;
        return 0;
    }

    IHostMemory *modelStream = myengine->serialize();
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    modelStream->destroy();
    myengine->destroy();
    network->destroy();
    parser->destroy();
    std::cout << "[TRTEngine] convert onnx model to TensorRT engine model successfully!" << std::endl;
    return 0;
}

//int onnx2engine_enc(std::string onnx_filepath_enc, std::string engine_filepath, int batchsize, int imagesize, int type)
//{
//    // create builder
//    TRTLogger logger;
//    IBuilder *builder = createInferBuilder(logger);
//
//    // create network
//    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
//
//    // create onnxparser
//    auto parser = nvonnxparser::createParser(*network, logger);
//
//    std::vector<uint8_t> file_data_enc = Enutils::load_file(onnx_filepath_enc);
//    AES aes;
//    std::vector<uint8_t> file_data;
//    aes.ecbEncrypt(file_data_enc, file_data, aes.getEncryptKey(), 0);
//
//    size_t size = file_data.size();
//    uint8_t *buffer = file_data.data();
//    std::cout << "[TRTEngine] decode buffer size: {}" << size << std::endl;
//
//    bool ret = parser->parse(buffer, size);
//    if (!ret)
//    {
//        parser->destroy();
//        builder->destroy();
//        std::cerr << "[TRTEngine](ERROR) parse onnx model failed!";
//        return -1;
//    }
//
//    for (int i = 0; i < parser->getNbErrors(); ++i)
//    {
//        std::cout << parser->getError(i)->desc() << std::endl;
//    }
//    std::cout << "[TRTEngine] tensorRT load onnx model sucessful!";
//
//    // builder->setMaxBatchSize(batchsize);
//
//    IBuilderConfig *config = builder->createBuilderConfig();
//    // config->setMaxWorkspaceSize(1 * (1 << 30));
//    //  config->setFlag(BuilderFlag::kGPU_FALLBACK);
//    //	config->setFlag(BuilderFlag::kSTRICT_TYPES);
//    if (type == 1)
//    {
//        config->setFlag(nvinfer1::BuilderFlag::kFP16);
//    }
//    if (type == 2)
//    {
//        config->setFlag(nvinfer1::BuilderFlag::kINT8);
//    }
//
//    IOptimizationProfile *profile = builder->createOptimizationProfile();
//    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(batchsize, 3, imagesize, imagesize));
//    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(batchsize, 3, imagesize, imagesize));
//    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(batchsize, 3, imagesize, imagesize));
//    config->addOptimizationProfile(profile);
//
//    ICudaEngine *myengine = builder->buildEngineWithConfig(*network, *config);
//    std::cout << "[TRTEngine] try to save engine file now" << std::endl;
//    std::ofstream p(engine_filepath, std::ios::binary);
//    if (!p)
//    {
//        std::cerr << "[TRTEngine](ERROR) could not open plan output file" << std::endl;
//        return 0;
//    }
//
//    IHostMemory *modelStream = myengine->serialize();
//    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
//
//    // delete[] buffer;
//    modelStream->destroy();
//    myengine->destroy();
//    network->destroy();
//    parser->destroy();
//    std::cout << "[TRTEngine] convert onnx model to TensorRT engine model successfully!" << std::endl;
//
//    return 0;
//}

int generate_engine_model(std::string &engine_file, int batch_size, int infer_size, int mode)
{
    //_access 若文件存在，则返回0
    std::cout << "[TRTEngine] engine path:" << engine_file << std::endl;
    if (_access(engine_file.c_str(), 0) == 0)
    {
        std::cout << "[TRTEngine] engine file exists!" << std::endl;
        return 0;
    }

    std::string onnx_enc_file;
    std::string onnx_file;
    if (engine_file.find(".engine") == std::string::npos)
    {
        std::cerr<< "[TRTEngine](ERROR) engine path parse error!" << std::endl;
        return -1;
    }
    else
    {
        size_t pos = engine_file.find(".engine");
        onnx_enc_file = engine_file.substr(0, pos);
        onnx_enc_file += ".onnx.enc";
        onnx_file = engine_file.substr(0, pos);
        onnx_file += ".onnx";
    }
    if (_access(onnx_file.c_str(), 0) == 0) {
        std::cout<< "[TRTEngine] engine file not exists, generate engine model (with onnx)..." << std::endl;
        onnx2engine(onnx_file, engine_file, batch_size, infer_size, mode);
    }
    else {
        std::cerr<< "[TRTEngine](ERROR) can not find *.onnx.enc or *.onnx file!" << std::endl;
        std::cerr<< "[TRTEngine](ERROR) please put the model.onnx.enc/model.onnx in the right place!"<< std::endl;
        return -1;
    }
    
    std::cout << "[TRTEngine] generate_engine_model successful." << std::endl;
    return 0;
}

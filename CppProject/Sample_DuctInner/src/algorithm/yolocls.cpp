#include "yolocls.h"

int YOLOClassify::model_init(const std::string model_path, const int gpu_id)
{
    spdlog::trace("YOLOClassify::model_init");
    cudaSetDevice(DEVICE);
    this->trt = new YOLOClsTRTContext();
    TRTLogger logger;

    auto engine_data = Enutils::load_engine_file(model_path);
    auto runtime = nvinfer1::createInferRuntime(logger);
    spdlog::trace("engine_data:{}", engine_data.size());
    trt->engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (trt->engine == nullptr)
    {
        spdlog::error("YOLOClassify deserialize cuda engine failed, engine not found.");
        runtime->destroy();
        return -1;
    };

    int binding_num = trt->engine->getNbBindings();
    if (binding_num != 2)
    {
        spdlog::error("yolo classify onnx export error，must have only one input and one output, your binding_num={}", binding_num - 1);
        return -1;
    };

    inputTensorShape.clear();
    outputTensorShape.clear();
    for (int i = 0; i < binding_num; i++)
    {
        std::string msg = "";
        nvinfer1::Dims dims = trt->engine->getBindingDimensions(i);
        int ndims = dims.nbDims;
        msg = msg + "binding-" + std::to_string(i) + ":";
        for (int d = 0; d < dims.nbDims; d++)
        {
            int size = dims.d[d];
            msg = msg + " " + std::to_string(size);
            if (i == 0)
                inputTensorShape.push_back(size);
            else
                outputTensorShape.push_back(size);
        }
        spdlog::info(msg);
    }
    try
    {
        trt->input_batch = inputTensorShape[0];
        // int input_channel = inputTensorShape[1];
        trt->input_height = inputTensorShape[2];
        trt->input_width = inputTensorShape[3];

        trt->input_numel = Enutils::vectorProduct(inputTensorShape);
        trt->output_numel = Enutils::vectorProduct(outputTensorShape);
        //spdlog::trace("input_numel:{} output_numel:{}", trt->input_numel, trt->output_numel);
        trt->stream = nullptr;
        checkRuntime(cudaStreamCreate(&trt->stream));
        trt->context = trt->engine->createExecutionContext();

        trt->input_data_host = nullptr;
        trt->input_data_device = nullptr;
        checkRuntime(cudaMallocHost(&(trt->input_data_host), trt->input_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&(trt->input_data_device), trt->input_numel * sizeof(float)));

        trt->output_data_device = nullptr;
        trt->output_data_host = nullptr;
        checkRuntime(cudaMalloc(&(trt->output_data_device), trt->output_numel * sizeof(float)));
        checkRuntime(cudaMallocHost(&trt->output_data_host, trt->output_numel * sizeof(float)));
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Model Init ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

void YOLOClassify::prepare_data_cpu(
    std::vector<cv::Mat> images, float *input_data_host, int infer_w, int infer_h, bool is_rgb, std::vector<float> mean_value, std::vector<float> std_value)
{
    spdlog::trace("YOLOClassify::prepare_data_cpu");
    for (int i = 0; i < images.size(); i++)
    {
        cv::Mat image = images[i];
        int height = image.rows;
        int width = image.cols;

        float ratio = std::min(infer_w * 1.0 / width, infer_h * 1.0 / height);
        int resize_w = int(width * ratio);
        int resize_h = int(height * ratio);

        cv::Mat resize_image;
        cv::resize(image, resize_image, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat pad_image(infer_h, infer_w, CV_8UC3, cv::Scalar(0, 0, 0));
        int pad_left = (infer_w - resize_image.cols) / 2;
        int pad_top = (infer_h - resize_image.rows) / 2;
        cv::Rect pad_rect = cv::Rect(pad_left, pad_top, resize_image.cols, resize_image.rows);
        resize_image.copyTo(pad_image(pad_rect));

        if (is_rgb)
            cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);

        int image_area = resize_image.cols * resize_image.rows;

        unsigned char *pimage = resize_image.data;

        float *phost_b = input_data_host + image_area * 0 + image_area * 3 * i;
        float *phost_g = input_data_host + image_area * 1 + image_area * 3 * i;
        float *phost_r = input_data_host + image_area * 2 + image_area * 3 * i;
        for (int j = 0; j < image_area; ++j, pimage += 3)
        {
            *phost_b++ = (pimage[0] - mean_value[0]) / std_value[0];
            *phost_g++ = (pimage[1] - mean_value[1]) / std_value[1];
            *phost_r++ = (pimage[2] - mean_value[2]) / std_value[2];
        }
    }
}

void YOLOClassify::trt_engine_infer(float *input_data_device, float *output_data_device, cudaStream_t stream, nvinfer1::IExecutionContext *execution_context)
{
    float *bindings[] = { input_data_device, output_data_device };
    bool success = execution_context->enqueueV2((void **)bindings, (cudaStream_t)stream, nullptr);
}

//static void getBestClassInfo(float *ptr, const int &numClasses, float &bestConf, int &bestClassId)
//{
//    for (int i = 0; i < numClasses; i++)
//    {
//        if (ptr[i] > bestConf)
//        {
//            bestConf = ptr[i];
//            bestClassId = i;
//        }
//    }
//}

static void softmax_array(float *src_arr, float *dst_arr, int length)
{
    float score_sum = 0;
    for (int i = 0; i < length; i++)
    {
        dst_arr[i] = *(src_arr + i);
        dst_arr[i] = exp(dst_arr[i]);
        score_sum += dst_arr[i];
    }

    for (int i = 0; i < length; i++)
    {
        dst_arr[i] /= score_sum;
    }
}

int YOLOClassify::postprocess_V1(cv::Mat &feat, std::vector<float> &scores, std::vector<int> &cls_ids)
{
    int size0 = feat.size[0];
    int size1 = feat.size[1];
    // std::cout << feat.size << std::endl;
    for (size_t i = 0; i < size0; i++)
    {
        float* feat_blob = (float*)feat.data + i * size1;
        // 动态分配足够大的空间，而不是固定的10个元素
        std::vector<float> arrs2(size1, 0.0f);
        // 或者使用：float* arrs2 = new float[size1](); 
        softmax_array(feat_blob, arrs2.data(), size1);
        auto max_it = std::max_element(arrs2.begin(), arrs2.end());
        int index = std::distance(arrs2.begin(), max_it);
        float score = arrs2[index];
        scores.push_back(score);
        cls_ids.push_back(index);
    }
    return 0;
}

int YOLOClassify::classify_V1(std::vector<cv::Mat> &images, std::vector<float> &scores, std::vector<int> &cls_ids, ConfigModel &config_model)
{
    if (images.size() < 1)
        return -1;

    int infer_w = trt->input_width;
    int infer_h = trt->input_height;
    bool is_rgb = config_model.is_rgb;
    std::vector<float> means = config_model.means;
    std::vector<float> stds = config_model.stds;

    int batchSize = trt->input_batch;
    int num_batches = std::ceil(images.size() * 1.0 / batchSize);

    scores.clear();
    cls_ids.clear();
    for (size_t i = 0; i < num_batches; i++)
    {
        std::vector<cv::Mat> curBatchs;
        for (size_t ii = 0; ii < batchSize; ii++)
        {
            int index = i * batchSize + ii;
            index = index < images.size() ? index : images.size() - 1;
            curBatchs.push_back(images[index]);
        }

        // printf("%s %d input_data_host:%d input_data_device:%d\n",__FUNCTION__, __LINE__, trt->input_data_host, trt->input_data_device);
        prepare_data_cpu(curBatchs, trt->input_data_host, trt->input_width, trt->input_height, is_rgb, means, stds);

        checkRuntime(cudaMemcpyAsync((void *)trt->input_data_device, (void *)trt->input_data_host, (size_t)trt->input_numel * sizeof(float),
            cudaMemcpyHostToDevice, (cudaStream_t)trt->stream));
        trt_engine_infer(trt->input_data_device, trt->output_data_device, trt->stream, trt->context);
        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output_data_host, (void *)trt->output_data_device, sizeof(float) * trt->output_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaStreamSynchronize(trt->stream));

        cv::Mat feat = cv::Mat(outputTensorShape, CV_32F, trt->output_data_host);
        // std::cout << feat.size[0] << " " << feat.size[1] << " " << feat.size[2] << std::endl;
        std::vector<float> scores_i = {};
        std::vector<int> cls_ids_i = {};
        postprocess_V1(feat, scores_i, cls_ids_i);

        scores.insert(scores.end(), scores_i.begin(), scores_i.end());
        cls_ids.insert(cls_ids.end(), cls_ids_i.begin(), cls_ids_i.end());
    }

    return 0;
}

int YOLOClassify::postprocess_V2(cv::Mat &feat, std::vector<float> &scores, std::vector<int> &cls_ids, float conf)
{
    int size0 = feat.size[0];
    int size1 = feat.size[1];
    // std::cout << feat.size << std::endl;
    for (size_t i = 0; i < size0; i++)
    {
        float* feat_blob = (float*)feat.data + i * size1;
        // 动态分配足够大的空间，而不是固定的10个元素
        std::vector<float> arrs2(size1, 0.0f);
        // 或者使用：float* arrs2 = new float[size1](); 
        softmax_array(feat_blob, arrs2.data(), size1);
        for (size_t j = 0; j < size1; j++)
        {
            auto max_it = std::max_element(arrs2.begin(), arrs2.end());
            int index = std::distance(arrs2.begin(), max_it);
            float score = arrs2[index];
            if (score >= conf)
            {
                scores.push_back(score);
                cls_ids.push_back(j);
            }
        } 
    }


    return 0;
}

int YOLOClassify::classify_V2(std::vector<cv::Mat> &images, std::vector<float> &scores, std::vector<int> &cls_ids, ConfigModel &config_model)
{
    if (images.size() < 1)
        return -1;

    int infer_w = trt->input_width;
    int infer_h = trt->input_height;
    bool is_rgb = config_model.is_rgb;
    std::vector<float> means = config_model.means;
    std::vector<float> stds = config_model.stds;

    int batchSize = trt->input_batch;
    int num_batches = std::ceil(images.size() * 1.0 / batchSize);

    scores.clear();
    cls_ids.clear();
    for (size_t i = 0; i < num_batches; i++)
    {
        std::vector<cv::Mat> curBatchs;
        for (size_t ii = 0; ii < batchSize; ii++)
        {
            int index = i * batchSize + ii;
            index = index < images.size() ? index : images.size() - 1;
            curBatchs.push_back(images[index]);
        }

        // printf("%s %d input_data_host:%d input_data_device:%d\n",__FUNCTION__, __LINE__, trt->input_data_host, trt->input_data_device);
        prepare_data_cpu(curBatchs, trt->input_data_host, trt->input_width, trt->input_height, is_rgb, means, stds);

        checkRuntime(cudaMemcpyAsync((void *)trt->input_data_device, (void *)trt->input_data_host, (size_t)trt->input_numel * sizeof(float),
            cudaMemcpyHostToDevice, (cudaStream_t)trt->stream));
        trt_engine_infer(trt->input_data_device, trt->output_data_device, trt->stream, trt->context);
        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output_data_host, (void *)trt->output_data_device, sizeof(float) * trt->output_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaStreamSynchronize(trt->stream));

        cv::Mat feat = cv::Mat(outputTensorShape, CV_32F, trt->output_data_host);
        // std::cout << feat.size[0] << " " << feat.size[1] << " " << feat.size[2] << std::endl;
        std::vector<float> scores_i;
        std::vector<int> cls_ids_i;
        postprocess_V2(feat, scores_i, cls_ids_i, config_model.conf_threshold);

        scores.insert(scores.end(), scores_i.begin(), scores_i.end());
        cls_ids.insert(cls_ids.end(), cls_ids_i.begin(), cls_ids_i.end());
    }

    return 0;
}

void YOLOClassify::cuda_free()
{
    checkRuntime(cudaStreamDestroy(trt->stream));

    checkRuntime(cudaFreeHost(trt->input_data_host));
    checkRuntime(cudaFree(trt->input_data_device));

    checkRuntime(cudaFree(trt->output_data_device));
    checkRuntime(cudaFreeHost(trt->output_data_host));

    delete trt->context;
    delete trt->engine;
}

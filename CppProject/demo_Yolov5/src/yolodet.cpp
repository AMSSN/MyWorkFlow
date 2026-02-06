#include "yolodet.h"

int YOLODetector::model_init(const std::string model_path, const int gpu_id)
{
    LOG(TRACE) << "model_init" << " " << __LINE__;

    cudaSetDevice(DEVICE);

    this->trt = new YOLODetTRTContext();

    TRTLogger logger;

    auto engine_data = enutils::load_engine_file(model_path);
    auto runtime = nvinfer1::createInferRuntime(logger);

    std::cout << "engine_data:" << engine_data.size() << std::endl;
    trt->engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    std::cout << "Model Init Start" << std::endl;
    if (trt->engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        std::cout << "[AI Detection]: Deserialize cuda engine failed, engine not found." << std::endl;
        runtime->destroy();
        return -1;
    };

    int binding_num = trt->engine->getNbBindings();

    if (binding_num != 2)
    {
        printf("onnx export error，must have only one input and one output，you have：%d output.\n", binding_num - 1);
        return -1;
    };

    inputTensorShape.clear();
    outputTensorShape.clear();
    for (int i = 0; i < binding_num; i++)
    {
        nvinfer1::Dims dims = trt->engine->getBindingDimensions(i);
        printf("binding %d: ", i);
        for (int d = 0; d < dims.nbDims; d++)
        {
            int size = dims.d[d];
            printf("%d ", size);
            if (i == 0)
                inputTensorShape.push_back(size);
            else
                outputTensorShape.push_back(size);
        }
        printf("\n");
    }

    try
    {
        trt->input_batch = inputTensorShape[0];
        trt->input_height = inputTensorShape[2];
        trt->input_width = inputTensorShape[3];

        trt->input_numel = enutils::vectorProduct(inputTensorShape);
        trt->output_numel = enutils::vectorProduct(outputTensorShape);
        printf("input_numel:%d output_numel:%d\n", trt->input_numel, trt->output_numel);

        trt->stream = nullptr;
        checkRuntime(cudaStreamCreate(&trt->stream));
        trt->context = trt->engine->createExecutionContext();

        trt->input_data_host = nullptr;
        trt->input_data_device = nullptr;
        checkRuntime(cudaMallocHost(&(trt->input_data_host), (trt->input_numel + 1) * sizeof(float)));
        checkRuntime(cudaMalloc(&(trt->input_data_device), (trt->input_numel + 1) * sizeof(float)));

        trt->output_data_device = nullptr;
        trt->output_data_host = nullptr;
        checkRuntime(cudaMalloc(&(trt->output_data_device), (trt->output_numel + 1) * sizeof(float)));
        checkRuntime(cudaMallocHost(&(trt->output_data_host), (trt->output_numel + 1) * sizeof(float)));
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "[AI Detection]: Model Init ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

void YOLODetector::prepare_data_cpu(
    std::vector<cv::Mat> images, float *input_data_host, int infer_w, int infer_h, bool is_rgb, std::vector<float> mean_value, std::vector<float> std_value)
{
    LOG(TRACE) << "prepare_data_cpu" << " " << __LINE__;

    for (size_t i = 0; i < images.size(); i++)
    {
        cv::Mat image = images[i];
        int height = image.rows;
        int width = image.cols;

        float ratio = min(infer_w * 1.0 / width, infer_h * 1.0 / height);
        int resize_w = int(width * ratio);
        int resize_h = int(height * ratio);

        cv::Mat resize_image;
        cv::resize(image, resize_image, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat pad_image(infer_h, infer_w, CV_8UC3, cv::Scalar(0, 0, 0));
        resize_image.copyTo(pad_image(cv::Rect(0, 0, resize_image.cols, resize_image.rows)));

        if (is_rgb)
            cv::cvtColor(pad_image, pad_image, cv::COLOR_BGR2RGB);

        int image_area = pad_image.cols * pad_image.rows;

        unsigned char *pimage = pad_image.data;

        float *phost_b = input_data_host + image_area * 0 + image_area * 3 * i;
        float *phost_g = input_data_host + image_area * 1 + image_area * 3 * i;
        float *phost_r = input_data_host + image_area * 2 + image_area * 3 * i;
        for (size_t j = 0; j < image_area; ++j, pimage += 3)
        {
            *phost_b++ = (pimage[0] - mean_value[0]) / std_value[0];
            *phost_g++ = (pimage[1] - mean_value[1]) / std_value[1];
            *phost_r++ = (pimage[2] - mean_value[2]) / std_value[2];
        }
    }
}

void YOLODetector::trt_engine_infer(float *input_data_device, float *output_data_device, cudaStream_t stream, nvinfer1::IExecutionContext *execution_context)
{
    LOG(TRACE) << "trt_engine_infer" << " " << __LINE__;

    float *bindings[] = { input_data_device, output_data_device };
    bool success = execution_context->enqueueV2((void **)bindings, (cudaStream_t)stream, nullptr);
}

static void getBestClassInfo(float *ptr, const int &numClasses, float &bestConf, int &bestClassId)
{
    for (size_t i = 0; i < numClasses; i++)
    {
        if (ptr[i] > bestConf)
        {
            bestConf = ptr[i];
            bestClassId = i;
        }
    }
}

int YOLODetector::postprocess(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres, float nms_thres, int postproc_type)
{
    LOG(TRACE) << "postprocess" << " " << __LINE__;

    if (postproc_type == 1)
        postprocess_v5(feat, vec_boxes, conf_thres, nms_thres); // yolov5的后处理方式
    else if (postproc_type == 2)
        postprocess_v8(feat, vec_boxes, conf_thres, nms_thres); // yolov8的后处理方式
    else if (postproc_type == 3)
        postprocess_v10(feat, vec_boxes, conf_thres); // yolov10的后处理方式
    else
        return -1;

    return 0;
}

int YOLODetector::postprocess_v5(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres, float nms_thres)
{
    LOG(TRACE) << "postprocess_v5" << " " << __LINE__;

    int size0 = feat.size[0];
    int size1 = feat.size[1];
    int size2 = feat.size[2];

    int numClasses = size2 - 5;
    int elementsInBatch = size1 * size2;
    // printf("dims:%d size0:%d size1:%d size2:%d numClasses:%d\n", feat.dims, size0, size1, size2, numClasses);

    for (size_t m = 0; m < size0; m++)
    {
        float *feat_blob = (float *)feat.data + m * elementsInBatch;

        std::vector<Bbox> boxes;

        for (size_t n = 0; n < size1; n++)
        {
            float *feat_ptr = feat_blob + n * size2;
            float clsConf = feat_ptr[4];
            if (clsConf > conf_thres)
            {
                float cx = feat_ptr[0];
                float cy = feat_ptr[1];
                float bw = feat_ptr[2];
                float bh = feat_ptr[3];
                int x1 = (int)(cx - bw / 2);
                int y1 = (int)(cy - bh / 2);
                int x2 = (int)(cx + bw / 2);
                int y2 = (int)(cy + bh / 2);

                float objConf = 0.f;
                int classId = -1;
                getBestClassInfo(feat_ptr + 5, numClasses, objConf, classId);

                float confidence = clsConf * objConf;

                Bbox box;
                box.x1 = x1;
                box.y1 = y1;
                box.x2 = x2;
                box.y2 = y2;
                box.score = confidence;
                box.label = classId;

                boxes.push_back(box);
            }
        }
        enutils::NMS(boxes, nms_thres);
        vec_boxes.push_back(boxes);
    }

    return 0;
}

int YOLODetector::postprocess_v8(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres, float nms_thres)
{
    LOG(TRACE) << "postprocess_v8" << " " << __LINE__;

    int size0 = feat.size[0];
    int size1 = feat.size[1]; // 14
    int size2 = feat.size[2]; // 33600
    // printf("dims:%d size0:%d size1:%d size2:%d\n", feat.dims, size0, size1, size2);

    int numClasses = size1 - 4;
    int elementsInBatch = size1 * size2;
    // printf("numClasses:%d elementsInBatch:%d\n", numClasses, elementsInBatch);

    for (int m = 0; m < size0; m++)
    {
        cv::Mat feat1 = cv::Mat(size1, size2, CV_32F, (float *)feat.data + m * elementsInBatch);
        cv::Mat feat1_t = feat1.t();
        float *feat_blob = (float *)feat1_t.data;

        std::vector<Bbox> boxes;
        for (size_t n = 0; n < size2; n++)
        {
            float *feat_ptr = feat_blob + n * size1;

            float objConf = 0.f;
            int classId = -1;
            getBestClassInfo(feat_ptr + 4, numClasses, objConf, classId);
            if (objConf > conf_thres)
            {
                float x = feat_ptr[0];
                float y = feat_ptr[1];
                float w = feat_ptr[2];
                float h = feat_ptr[3];

                int left = int(x - 0.5 * w);
                int top = int(y - 0.5 * h);
                int width = int(w);
                int height = int(h);
                float confidence = objConf;

                Bbox box;
                box.x1 = left;
                box.y1 = top;
                box.x2 = left + width;
                box.y2 = top + height;
                box.score = confidence;
                box.label = classId;

                boxes.push_back(box);
            }
        }
        enutils::NMS(boxes, nms_thres);
        vec_boxes.push_back(boxes);
    }

    return 0;
}

int YOLODetector::postprocess_v10(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres)
{
    LOG(TRACE) << "postprocess_v10" << " " << __LINE__;

    int size0 = feat.size[0];
    int size1 = feat.size[1];
    int size2 = feat.size[2];
    int elementsInBatch = size1 * size2;
    // printf("dims:%d size0:%d size1:%d size2:%d\n", feat.dims, size0, size1, size2);

    for (int m = 0; m < size0; m++)
    {
        float *feat_blob = (float *)feat.data + m * elementsInBatch;

        std::vector<Bbox> boxes;

        for (size_t n = 0; n < size1; n++)
        {
            float *feat_ptr = feat_blob + n * size2;
            float clsConf = feat_ptr[4];
            if (clsConf > conf_thres)
            {
                float objConf = 0.f;
                int classId = -1;
                classId = feat_ptr[5];
                float confidence = clsConf * objConf;
                Bbox box;
                box.x1 = feat_ptr[0];
                box.y1 = feat_ptr[1];
                box.x2 = feat_ptr[2];
                box.y2 = feat_ptr[3];
                box.score = confidence;
                box.label = classId;
                boxes.push_back(box);
            }
        }
        vec_boxes.push_back(boxes);
    }

    return 0;
}

int YOLODetector::detect1(std::vector<cv::Mat> &images, std::vector<std::vector<Bbox>> &vec_boxes, ConfigModel &config_model)
{
    LOG(TRACE) << "detect1" << " " << __LINE__;
    if (images.size() < 1)
        return -1;

    int infer_w = trt->input_width;
    int infer_h = trt->input_height;
    bool is_rgb = config_model.is_rgb;
    float score_thresh = config_model.conf_threshold;
    float iou_thresh = config_model.nms_threshold;
    int postproc_type = config_model.postproc_type;
    std::vector<float> means = config_model.means;
    std::vector<float> stds = config_model.stds;

    int batchSize = trt->input_batch;
    int num_batches = std::ceil(images.size() * 1.0 / batchSize);

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

        std::vector<std::vector<Bbox>> vboxes_i;
        postprocess(feat, vboxes_i, score_thresh, iou_thresh, postproc_type);

        for (size_t ii = 0; ii < vboxes_i.size(); ii++)
        {
            int index = ii + i * batchSize;
            if (index >= images.size())
                break;
            cv::Mat image = images[index];
            int height = image.rows;
            int width = image.cols;

            std::vector<Bbox> boxes = vboxes_i[ii];
            float ratio = min(infer_w * 1.0 / width, infer_h * 1.0 / height);

            for (auto &box : boxes)
            {
                int x1 = int(box.x1 / ratio);
                int y1 = int(box.y1 / ratio);
                int x2 = int(box.x2 / ratio);
                int y2 = int(box.y2 / ratio);
                box.x1 = enutils::clip(x1, 0, width - 1);
                box.x2 = enutils::clip(x2, x1 + 1, width);
                box.y1 = enutils::clip(y1, 0, height - 1);
                box.y2 = enutils::clip(y2, y1 + 1, height);
            }
            vec_boxes.push_back(boxes);
        }
    }

    // checkRuntime(cudaFreeHost(trt->input_data_host));
    // checkRuntime(cudaFree(trt->input_data_device));
    // checkRuntime(cudaFree(trt->output_data_device));
    // checkRuntime(cudaFreeHost(trt->output_data_host));

    return 0;
}

int YOLODetector::detect2(cv::Mat &image, std::vector<Bbox> &final_boxes, ConfigModel &config_model)
{
    LOG(TRACE) << "detect2" << " " << __LINE__;

    int infer_w = trt->input_width;
    int infer_h = trt->input_height;
    int batchSize = trt->input_batch;

    int crop_size = config_model.crop_size;
    int stride = config_model.stride;

    float conf_thres = config_model.conf_threshold;
    float iou_thres = config_model.nms_threshold;
    int postproc_type = config_model.postproc_type;
    int is_rgb = config_model.is_rgb;
    std::vector<float> means = config_model.means;
    std::vector<float> stds = config_model.stds;

    int overlap = crop_size - stride;

    int height = image.rows;
    int width = image.cols;

    std::vector<cv::Mat> crop_imgs;
    std::vector<std::pair<int, int>> coords;

    int nx = ceil(abs(width - overlap) * 1.0 / stride);
    int ny = ceil(abs(height - overlap) * 1.0 / stride);
    int tblocks = nx * ny;
    // printf("width:%d height:%d nx:%d ny:%d\n", width,height,nx, ny);

    int index = 0;
    for (int by = 0; by < ny; by++)
    {
        for (int bx = 0; bx < nx; bx++)
        {
            index = by * nx + bx;
            int blk_x1 = stride * bx;
            int blk_y1 = stride * by;
            int blk_x2 = overlap + stride * (bx + 1);
            int blk_y2 = overlap + stride * (by + 1);
            blk_x2 = enutils::clip(blk_x2, blk_x1, width);
            blk_y2 = enutils::clip(blk_y2, blk_y1, height);
            int blk_h = blk_y2 - blk_y1;
            int blk_w = blk_x2 - blk_x1;
            if (blk_h < 1 || blk_w < 1)
                continue;

            cv::Mat crop_img = image(cv::Rect(blk_x1, blk_y1, blk_w, blk_h));
            if (blk_h < crop_size || blk_w < crop_size)
            {
                cv::Mat pad_image = cv::Mat::zeros(crop_size, crop_size, CV_8UC3);
                crop_img.copyTo(pad_image(cv::Rect(0, 0, blk_w, blk_h)));
                crop_img = pad_image;
            }

            crop_imgs.push_back(crop_img);
            coords.push_back(std::pair<int, int>(blk_x1, blk_y1));
        }
    }

    // printf("crop_imgs.size:%d coords.size:%d\n", crop_imgs.size(), coords.size());
    if (crop_imgs.size() != coords.size())
        return -1;

    int num_batches = ceil(crop_imgs.size() * 1.0 / batchSize);
    // float ratio = crop_size * 1.0 / infer_w;
    std::vector<Bbox> total_boxes;

    for (int i = 0; i < num_batches; i++)
    {
        std::vector<cv::Mat> curBatchs;
        for (int ii = 0; ii < batchSize; ii++)
        {
            int index = i * batchSize + ii;
            index = index < crop_imgs.size() ? index : crop_imgs.size() - 1;
            curBatchs.push_back(crop_imgs[index]);
        }

        prepare_data_cpu(curBatchs, trt->input_data_host, trt->input_width, trt->input_height, is_rgb, means, stds);
        checkRuntime(cudaMemcpyAsync((void *)trt->input_data_device, (void *)trt->input_data_host, (size_t)trt->input_numel * sizeof(float),
            cudaMemcpyHostToDevice, (cudaStream_t)trt->stream));
        trt_engine_infer(trt->input_data_device, trt->output_data_device, trt->stream, trt->context);

        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output_data_host, (void *)trt->output_data_device, sizeof(float) * trt->output_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaStreamSynchronize(trt->stream));

        cv::Mat feat = cv::Mat(outputTensorShape, CV_32F, trt->output_data_host);

        std::vector<std::vector<Bbox>> vboxes_i;
        postprocess(feat, vboxes_i, conf_thres, iou_thres, postproc_type);

        for (int ii = 0; ii < vboxes_i.size(); ii++)
        {
            int index = ii + i * batchSize;
            if (index >= crop_imgs.size())
                break;
            int blk_x1 = coords[index].first;
            int blk_y1 = coords[index].second;
            cv::Mat crop_img = crop_imgs[index];

            std::vector<Bbox> boxes = vboxes_i[ii];
            float ratio = min(infer_w * 1.0 / crop_size, infer_h * 1.0 / crop_size);

            for (auto &box : boxes)
            {
                int x1 = int(box.x1 / ratio + blk_x1);
                int y1 = int(box.y1 / ratio + blk_y1);
                int x2 = int(box.x2 / ratio + blk_x1);
                int y2 = int(box.y2 / ratio + blk_y1);
                // float score = box.score;
                // printf("x1:%d y1:%d x2:%d y2:%d label:%d score:%.2f\n", x1, y1, x2, y2, label, score);

                box.x1 = enutils::clip(x1, 0, width - 1);
                box.x2 = enutils::clip(x2, x1 + 1, width);
                box.y1 = enutils::clip(y1, 0, height - 1);
                box.y2 = enutils::clip(y2, y1 + 1, height);
                total_boxes.push_back(box);
            }
        }

        memset(trt->output_data_host, 0, trt->output_numel * sizeof(float));
        cudaMemset(trt->output_data_device, 0, trt->output_numel * sizeof(float));
    }

    enutils::mergeOverlapOutputs(total_boxes, final_boxes, "Min", 0.001);

    crop_imgs.clear();
    coords.clear();

    return 0;
}

void YOLODetector::cuda_free()
{
    LOG(TRACE) << "cuda_free" << " " << __LINE__;

    checkRuntime(cudaStreamDestroy(trt->stream));

    checkRuntime(cudaFreeHost(trt->input_data_host));
    checkRuntime(cudaFree(trt->input_data_device));

    checkRuntime(cudaFree(trt->output_data_device));
    checkRuntime(cudaFreeHost(trt->output_data_host));

    delete trt->context;
    delete trt->engine;
    LOG(INFO) << "Detector: CUDA Context Free Finished";
}

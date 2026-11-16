#include "yoloseg.h"

int YOLOSegmentor::model_init(const std::string model_path, const int gpu_id)
{
    spdlog::trace("YOLOSegmentor::model_init");
    cudaSetDevice(gpu_id);
    this->trt = new YOLOSegTRTContext();
    TRTLogger logger;
    auto engine_data = Enutils::load_engine_file(model_path);
    auto runtime = nvinfer1::createInferRuntime(logger);
    // LOG(INFO) << "engine_data:" << engine_data.size();
    trt->engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    spdlog::trace("Yolo segment model_init Start");
    if (trt->engine == nullptr)
    {
        spdlog::error("YOLOSegmentor deserialize cuda engine failed, engine not found.");
        runtime->destroy();
        return -1;
    };

    int binding_num = trt->engine->getNbBindings();
    if (binding_num != 3)
    {
        spdlog::error("yolo InstSeg onnx export error，must have one input and 3 output");
        return -1;
    };

    std::vector<int> temp_out0_shapes;
    std::vector<int> temp_out1_shapes;

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
                input_tensor_shapes.push_back(size);
            else if (i == 1)
                temp_out0_shapes.push_back(size);
            else if (i == 2)
                temp_out1_shapes.push_back(size);
        }
        spdlog::info(msg);
    }

    // 在onnx转engine的过程中，两个output的顺序会颠倒，需要保证output0_tensor是1x56700x38，output1_tensor是1x32x240x240
    if (temp_out0_shapes.size() == 3 && temp_out1_shapes.size() == 4)
    {
        this->output_tensor_order = true;
        this->output0_tensor_shapes = temp_out0_shapes;
        this->output1_tensor_shapes = temp_out1_shapes;
    }
    else if (temp_out0_shapes.size() == 4 && temp_out1_shapes.size() == 3)
    {
        this->output_tensor_order = false;
        this->output0_tensor_shapes = temp_out1_shapes;
        this->output1_tensor_shapes = temp_out0_shapes;
    }
    else
    {
        spdlog::error("InstSeg output tensor order error!");
        return -1;
    }

    try
    {
        trt->input_batch = input_tensor_shapes[0];
        int input_channel = input_tensor_shapes[1];
        trt->input_height = input_tensor_shapes[2];
        trt->input_width = input_tensor_shapes[3];

        trt->input_numel = Enutils::vectorProduct(input_tensor_shapes);
        trt->output0_numel = Enutils::vectorProduct(output0_tensor_shapes);
        trt->output1_numel = Enutils::vectorProduct(output1_tensor_shapes);
        // LOG(INFO) << "input_numel:" << trt->input_numel << " output0_numel:" << trt->output0_numel << " trt->output1_numel:" << trt->output1_numel;

        _segChannels = output1_tensor_shapes[1];
        _segHeight = output1_tensor_shapes[2];
        _segWidth = output1_tensor_shapes[3];
        _classNum = output0_tensor_shapes[2] - _segChannels - 5;
        // LOG(INFO) << "_segChannels:" << _segChannels << " _segHeight:" << _segHeight << " _segWidth:" << _segWidth << " _classNum:" << _classNum;

        trt->stream = nullptr;
        checkRuntime(cudaStreamCreate(&trt->stream));
        trt->context = trt->engine->createExecutionContext();

        trt->input_data_host = nullptr;
        trt->input_data_device = nullptr;
        checkRuntime(cudaMallocHost(&(trt->input_data_host), trt->input_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&(trt->input_data_device), trt->input_numel * sizeof(float)));

        trt->output0_data_device = nullptr;
        trt->output0_data_host = nullptr;
        checkRuntime(cudaMalloc(&(trt->output0_data_device), trt->output0_numel * sizeof(float)));
        checkRuntime(cudaMallocHost(&trt->output0_data_host, trt->output0_numel * sizeof(float)));

        trt->output1_data_device = nullptr;
        trt->output1_data_host = nullptr;
        checkRuntime(cudaMalloc(&(trt->output1_data_device), trt->output1_numel * sizeof(float)));
        checkRuntime(cudaMallocHost(&trt->output1_data_host, trt->output1_numel * sizeof(float)));
    }
    catch (std::exception e)
    {
        spdlog::error("InstSeg Model Init ERROR: {}" , e.what());
        return -1;
    }

    return 0;
}

void YOLOSegmentor::prepare_data_cpu(
    std::vector<cv::Mat> images, float *input_data_host, int infer_w, int infer_h, bool is_rgb, std::vector<float> mean_value, std::vector<float> std_value)
{
    spdlog::trace("InstSeg::prepare_data_cpu");

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
        resize_image.copyTo(pad_image(cv::Rect(0, 0, resize_image.cols, resize_image.rows)));

        if (is_rgb)
            cv::cvtColor(pad_image, pad_image, cv::COLOR_BGR2RGB);

        int image_area = pad_image.cols * pad_image.rows;

        unsigned char *pimage = pad_image.data;

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

void YOLOSegmentor::trt_engine_infer(
    float *input_data_device, float *output0_data_device, float *output1_data_device, cudaStream_t stream, nvinfer1::IExecutionContext *execution_context)
{
    spdlog::trace("InstSeg::trt_engine_infer");
    float *bindings[] = { input_data_device, output0_data_device, output1_data_device };
    bool success = execution_context->enqueueV2((void **)bindings, (cudaStream_t)stream, nullptr);
}

int YOLOSegmentor::postprocess(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres, float nms_thres, int postproc_type)
{
    spdlog::trace("YOLOSegmentor::postprocess");
    spdlog::debug("YOLOSegmentor::postprocess type={}", postproc_type);
    //if (postproc_type == 1)
    //    postprocess_v5(out_feats, vec_segs, conf_thres, nms_thres); // yolov5的后处理方式
    //else if (postproc_type == 2)
    //    postprocess_v8(out_feats, vec_segs, conf_thres, nms_thres); // yolov8的后处理方式
    //else if (postproc_type == 3)
    //    postprocess_v10(out_feats, vec_segs, conf_thres); // yolov10的后处理方式
    //else
    //    return -1;

    postprocess_v5(out_feats, vec_segs, conf_thres, nms_thres); // yolov5的后处理方式
    return 0;
}

int YOLOSegmentor::postprocess_v5(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres, float nms_thres)
{
    spdlog::trace("InstSeg::postprocess");
    int net_height = output0_tensor_shapes[1];
    int net_width = output0_tensor_shapes[2];
    spdlog::trace("net_height={}, net_width={}", net_height, net_width);
    cv::Mat out_tensor = out_feats[0];
    cv::Mat out_mask_tensor = out_feats[1];

    int size0 = out_tensor.size[0];
    for (int index = 0; index < size0; index++)
    {
        float* pdata = (float*)out_tensor.data;

        std::vector<int> class_ids;                       // 结果id数组
        std::vector<float> confidences;                   // 结果每个id对应置信度数组
        std::vector<cv::Rect> boxes;                      // 每个id矩形框
        std::vector<std::vector<float>> picked_proposals; // output0[:,:, 5 + _className.size():net_width]===> for mask

        for (int r = 0; r < net_height; r++)
        {
            float box_score = pdata[4]; // box-confidence
            if (box_score >= conf_thres)
            {
                cv::Mat scores(1, _classNum, CV_32FC1, pdata + 5);
                cv::Point classIdPoint;
                double max_class_socre;
                cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                max_class_socre = (float)max_class_socre;
                if (max_class_socre >= conf_thres)
                {
                    std::vector<float> temp_proto(pdata + 5 + _classNum, pdata + net_width);
                    // printf("temp_proto.size:%d\n", temp_proto.size());
                    picked_proposals.push_back(temp_proto);
                    // rect [x,y,w,h]
                    float x = pdata[0];
                    float y = pdata[1];
                    float w = pdata[2];
                    float h = pdata[3];
                    int left = std::max(int(x - 0.5 * w + 0.5), 0);
                    int top = std::max(int(y - 0.5 * h + 0.5), 0);
                    int box_w = int(w + 0.5);
                    int box_h = int(h + 0.5);
                    // printf("left:%d top:%d box_w:%d box_h:%d\n", left, top, box_w, box_h);
                    class_ids.push_back(classIdPoint.x);
                    confidences.push_back(max_class_socre * box_score);
                    boxes.push_back(cv::Rect(left, top, box_w, box_h));
                }
            }
            pdata += net_width; // 下一行
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thres, nms_thres, nms_result);

        std::vector<std::vector<float>> temp_mask_proposals;
        cv::Rect holeImgRect(0, 0, trt->input_width, trt->input_height);
        spdlog::trace("trt->input_width={}, trt->input_height={}", trt->input_width, trt->input_height);
        std::vector<YoloMask> temp_output;
        for (int i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];
            YoloMask result;
            result.label = class_ids[idx];
            result.score = confidences[idx];
            result.box = boxes[idx] & holeImgRect;
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

        // MaskParams mask_params;
        std::vector<int> mask_shapes = { 1, output1_tensor_shapes[1], output1_tensor_shapes[2], output1_tensor_shapes[3] };
        int mask_protos_length = Enutils::vectorProduct(mask_shapes);

        cv::Mat mask_protos = cv::Mat(mask_shapes, CV_32F, (float*)out_mask_tensor.data + index * mask_protos_length);
        for (int i = 0; i < temp_mask_proposals.size(); ++i)
        {
            get_yolo_mask(cv::Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i]);
        }
        // printf("temp_output.size:%d\n", temp_output.size());
        spdlog::trace("temp_output.size:{}", temp_output.size());
        vec_segs.push_back(temp_output);
    }

    return 0;
}
int YOLOSegmentor::postprocess_v8(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres, float nms_thres)
{
    return 0;
}
int YOLOSegmentor::postprocess_v10(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres)
{
    return 0;
}




void YOLOSegmentor::get_yolo_mask(const cv::Mat &maskProposals, const cv::Mat &mask_protos, YoloMask&output_seg)
{
    spdlog::trace("InstSeg::get_yolo_mask");
    int seg_channels = _segChannels;
    int seg_width = _segWidth;
    int seg_height = _segHeight;
    int net_width = trt->input_width;
    int net_height = trt->input_height;
    float mask_threshold = 0.5;
    cv::Size src_img_shape = cv::Size(net_width, net_height);

    cv::Rect temp_rect = output_seg.box;
    // crop from mask_protos
    int rang_x = floor(temp_rect.x * 1.0 / net_width * seg_width);
    int rang_y = floor(temp_rect.y * 1.0 / net_height * seg_height);
    int rang_w = ceil((temp_rect.x + temp_rect.width) * 1.0 / net_width * seg_width) - rang_x;
    int rang_h = ceil((temp_rect.y + temp_rect.height) * 1.0 / net_height * seg_height) - rang_y;

    // 如果下面的 mask_protos(roi_rangs).clone()位置报错，说明你的output.box数据不对，或者矩形框就1个像素的，开启下面的注释部分防止报错。
    rang_w = std::max(rang_w, 1);
    rang_h = std::max(rang_h, 1);
    if (rang_x + rang_w > seg_width)
    {
        if (seg_width - rang_x > 0)
            rang_w = seg_width - rang_x;
        else
            rang_x -= 1;
    }
    if (rang_y + rang_h > seg_height)
    {
        if (seg_height - rang_y > 0)
            rang_h = seg_height - rang_y;
        else
            rang_y -= 1;
    }

    std::vector<cv::Range> roi_rangs;
    roi_rangs.push_back(cv::Range(0, 1));
    roi_rangs.push_back(cv::Range::all());
    roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
    roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

    // crop
    cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
    cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels, rang_w * rang_h });
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks_feature = matmul_res.reshape(1, { rang_h, rang_w });
    cv::Mat dest, mask;

    // sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor(net_width / seg_width * rang_x);
    int top = floor(net_height / seg_height * rang_y);
    int width = ceil(net_width / seg_width * rang_w);
    int height = ceil(net_height / seg_height * rang_h);

    cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
    mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
    output_seg.boxMask = mask;
}

int YOLOSegmentor::detect1(std::vector<cv::Mat> &images, std::vector<std::vector<YoloMask>> &vec_outsegs, ConfigModel &config_model)
{
    spdlog::trace("InstSeg::detect1");
    if (images.size() < 1)
        return -1;

    clock_t start33, end33;
    start33 = clock();

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

    for (int i = 0; i < num_batches; i++)
    {
        std::vector<cv::Mat> curBatchs;
        for (int ii = 0; ii < batchSize; ii++)
        {
            int index = i * batchSize + ii;
            index = index < images.size() ? index : images.size() - 1;
            curBatchs.push_back(images[index]);
        }

        prepare_data_cpu(curBatchs, trt->input_data_host, trt->input_width, trt->input_height, is_rgb, means, stds);
        checkRuntime(cudaMemcpyAsync((void *)trt->input_data_device, (void *)trt->input_data_host, (size_t)trt->input_numel * sizeof(float),
            cudaMemcpyHostToDevice, (cudaStream_t)trt->stream));
        // 调整顺序，保证output0是shape为3的数据流，output1是shape为4的数据流
        // std::cout << "output_tensor_order: " << output_tensor_order << std::endl;
        if (this->output_tensor_order)
            trt_engine_infer(trt->input_data_device, trt->output0_data_device, trt->output1_data_device, trt->stream, trt->context);
        else
            trt_engine_infer(trt->input_data_device, trt->output1_data_device, trt->output0_data_device, trt->stream, trt->context);

        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output0_data_host, (void *)trt->output0_data_device, sizeof(float) * trt->output0_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output1_data_host, (void *)trt->output1_data_device, sizeof(float) * trt->output1_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaStreamSynchronize(trt->stream));
        spdlog::trace("output0_tensor_shapes: ");
        for (auto size : output0_tensor_shapes)
            spdlog::trace(size+" ");
        cv::Mat feat0 = cv::Mat(output0_tensor_shapes, CV_32F, trt->output0_data_host);
        spdlog::trace("output1_tensor_shapes: ");
        for (auto size : output1_tensor_shapes)
            spdlog::trace(size + " ");
        // std::cout << std::endl;
        cv::Mat feat1 = cv::Mat(output1_tensor_shapes, CV_32F, trt->output1_data_host);

        std::vector<cv::Mat> out_feats;
        out_feats.push_back(feat0);
        out_feats.push_back(feat1);

        std::vector<std::vector<YoloMask>> vec_segs_i;
        this->postprocess(out_feats, vec_segs_i, score_thresh, iou_thresh, postproc_type);

        for (int ii = 0; ii < vec_segs_i.size(); ii++)
        {
            int index = ii + i * batchSize;
            if (index >= images.size())
                break;
            cv::Mat image = images[index];
            int height = image.rows;
            int width = image.cols;

            std::vector<YoloMask> output_segs = vec_segs_i[ii];
            float ratio = std::min(infer_w * 1.0 / width, infer_h * 1.0 / height);
            // printf("ratio:%f\n", ratio);

            for (auto &output_seg : output_segs)
            {
                // printf("box.x:%d box.y:%d box.w:%d box.h:%d\n", output_seg.box.x, output_seg.box.y, output_seg.box.width, output_seg.box.height);
                int x1 = int(output_seg.box.x / ratio);
                int y1 = int(output_seg.box.y / ratio);
                int x2 = int((output_seg.box.x + output_seg.box.width) / ratio);
                int y2 = int((output_seg.box.y + output_seg.box.height) / ratio);
                x1 = Enutils::clip(x1, 0, width - 1);
                x2 = Enutils::clip(x2, x1 + 1, width);
                y1 = Enutils::clip(y1, 0, height - 1);
                y2 = Enutils::clip(y2, y1 + 1, height);
                cv::Rect rect1 = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                output_seg.box = rect1;

                cv::Mat boxMask1;
                cv::resize(output_seg.boxMask, boxMask1, cv::Size(rect1.width, rect1.height), 0, 0, cv::INTER_NEAREST);
                output_seg.boxMask = boxMask1;
            }
            vec_outsegs.push_back(output_segs);
        }

        memset(trt->output0_data_host, 0, trt->output0_numel * sizeof(float));
        cudaMemset(trt->output0_data_device, 0, trt->output0_numel * sizeof(float));
        memset(trt->output1_data_host, 0, trt->output1_numel * sizeof(float));
        cudaMemset(trt->output1_data_device, 0, trt->output1_numel * sizeof(float));
    }
    end33 = clock();
    spdlog::trace("yoloseg detect1 time = {}s", double(end33 - start33) / CLOCKS_PER_SEC);
    return 0;
}

int YOLOSegmentor::detect2(cv::Mat &image, std::vector<YoloMask> &final_outsegs, ConfigModel &config_model)
{
    spdlog::trace("InstSeg::detect2");
    int infer_w = trt->input_width;
    int infer_h = trt->input_height;
    int batchSize = trt->input_batch;
    int crop_size = config_model.crop_size;
    int stride = config_model.stride;
    int postproc_type = config_model.postproc_type;
    float score_thresh = config_model.conf_threshold;
    float iou_thresh = config_model.nms_threshold;
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
            blk_x2 = Enutils::clip(blk_x2, blk_x1, width);
            blk_y2 = Enutils::clip(blk_y2, blk_y1, height);
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
    {
        spdlog::error("image sliding crop error!");
        return -1;
    }

    int num_batches = std::ceil(crop_imgs.size() * 1.0 / batchSize);
    std::vector<YoloMask> total_outsegs;

    for (int i = 0; i < num_batches; i++)
    {
        // printf("batch_id:%d\n", i);
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
        // 调整顺序，保证output0是shape为3的数据流，output1是shape为4的数据流
        // std::cout << "output_tensor_order: " << output_tensor_order << std::endl;
        if (this->output_tensor_order)
            trt_engine_infer(trt->input_data_device, trt->output0_data_device, trt->output1_data_device, trt->stream, trt->context);
        else
            trt_engine_infer(trt->input_data_device, trt->output1_data_device, trt->output0_data_device, trt->stream, trt->context);

        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output0_data_host, (void *)trt->output0_data_device, sizeof(float) * trt->output0_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaMemcpyAsync(
            (void *)trt->output1_data_host, (void *)trt->output1_data_device, sizeof(float) * trt->output1_numel, cudaMemcpyDeviceToHost, trt->stream));
        checkRuntime(cudaStreamSynchronize(trt->stream));

        // std::cout << "output0_tensor_shapes: ";
        // for (auto size : output0_tensor_shapes)
        //	std::cout << size << " ";
        // std::cout << std::endl;
        cv::Mat feat0 = cv::Mat(output0_tensor_shapes, CV_32F, trt->output0_data_host);

        // std::cout << "output1_tensor_shapes: ";
        // for (auto size : output1_tensor_shapes)
        //	std::cout << size << " ";
        // std::cout << std::endl;
        cv::Mat feat1 = cv::Mat(output1_tensor_shapes, CV_32F, trt->output1_data_host);

        std::vector<cv::Mat> out_feats;
        out_feats.push_back(feat0);
        out_feats.push_back(feat1);

        std::vector<std::vector<YoloMask>> vec_segs_i;
        this->postprocess(out_feats, vec_segs_i, score_thresh, iou_thresh, postproc_type);

        for (int ii = 0; ii < vec_segs_i.size(); ii++)
        {
            int index = ii + i * batchSize;
            if (index >= crop_imgs.size())
                break;
            int blk_x1 = coords[index].first;
            int blk_y1 = coords[index].second;
            cv::Mat crop_img = crop_imgs[index];

            std::vector<YoloMask> output_segs = vec_segs_i[ii];
            float ratio = std::min(infer_w * 1.0 / crop_size, infer_h * 1.0 / crop_size);
            // printf("ratio:%f\n", ratio);

            for (auto &output_seg : output_segs)
            {
                // printf("box.x:%d box.y:%d box.w:%d box.h:%d\n", output_seg.box.x, output_seg.box.y, output_seg.box.width, output_seg.box.height);
                int x1 = int(output_seg.box.x / ratio + blk_x1);
                int y1 = int(output_seg.box.y / ratio + blk_y1);
                int x2 = int((output_seg.box.x + output_seg.box.width) / ratio + blk_x1);
                int y2 = int((output_seg.box.y + output_seg.box.height) / ratio + blk_y1);
                x1 = Enutils::clip(x1, 0, width - 1);
                x2 = Enutils::clip(x2, x1 + 1, width);
                y1 = Enutils::clip(y1, 0, height - 1);
                y2 = Enutils::clip(y2, y1 + 1, height);
                cv::Rect rect1 = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                output_seg.box = rect1;

                cv::Mat boxMask1;
                cv::resize(output_seg.boxMask, boxMask1, cv::Size(rect1.width, rect1.height), 0, 0, cv::INTER_NEAREST);
                output_seg.boxMask = boxMask1;

                total_outsegs.push_back(output_seg);
            }
        }

        memset(trt->output0_data_host, 0, trt->output0_numel * sizeof(float));
        cudaMemset(trt->output0_data_device, 0, trt->output0_numel * sizeof(float));
        memset(trt->output1_data_host, 0, trt->output1_numel * sizeof(float));
        cudaMemset(trt->output1_data_device, 0, trt->output1_numel * sizeof(float));
    }

    crop_imgs.clear();
    coords.clear();

    return 0;
}

void YOLOSegmentor::cuda_free()
{
    spdlog::trace("InstSeg::cuda_free");

    checkRuntime(cudaStreamDestroy(trt->stream));

    checkRuntime(cudaFreeHost(trt->input_data_host));
    checkRuntime(cudaFree(trt->input_data_device));

    checkRuntime(cudaFree(trt->output0_data_device));
    checkRuntime(cudaFreeHost(trt->output0_data_host));

    checkRuntime(cudaFree(trt->output1_data_device));
    checkRuntime(cudaFreeHost(trt->output1_data_host));

    delete trt->context;
    delete trt->engine;
    spdlog::info("YOLOSegmentor: CUDA Context Free Finished");
}

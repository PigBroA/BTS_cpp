#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <libgen.h>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/torch.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <torch/nn/parallel/data_parallel.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <bts/densenet.h>
#include <bts/bts.h>
#include <glob/glob.h>

class CustomDatasetDepthEstimation : public torch::data::datasets::Dataset<CustomDatasetDepthEstimation> {
public:
    CustomDatasetDepthEstimation(const std::vector<std::pair<std::string, std::string>>& data, int imgWidth, int imgHeight) : data(data) {
        this->imgWidth = imgWidth;
        this->imgHeight = imgHeight;
    }

    torch::data::Example<> get(size_t index) {
        cv::Mat img = cv::imread(data[index].first, cv::IMREAD_UNCHANGED);
        cv::Mat depthGt = cv::imread(data[index].second, cv::IMREAD_UNCHANGED);
        cv::resize(img, img, cv::Size(this->imgWidth, this->imgHeight));
        cv::resize(depthGt, depthGt, cv::Size(this->imgWidth, this->imgHeight));
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        torch::Tensor r = torch::from_blob(channels[2].ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.485).div(0.229);
        torch::Tensor g = torch::from_blob(channels[1].ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.456).div(0.224);
        torch::Tensor b = torch::from_blob(channels[0].ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.406).div(0.225);
        torch::Tensor imgTensor = torch::cat({r, g, b}).view({3, imgHeight, imgWidth}).pin_memory();
        torch::Tensor depthGtTensor = torch::from_blob(depthGt.ptr(), {imgHeight, imgWidth}, torch::kInt16).to(torch::kFloat32).div(100.0).unsqueeze(0).pin_memory();
        return {imgTensor, depthGtTensor};
    }

    torch::optional<size_t> size() const {
        return data.size();
    }
private:
    std::vector<std::pair<std::string, std::string>> data;
    int imgWidth;
    int imgHeight;
};

int main(int argc, char** argv) {
    if(!torch::cuda::is_available()) {
        std::cerr << "check cuda!!!" << std::endl;
        exit(1);
    }

    torch::manual_seed(time(0));
    torch::cuda::manual_seed(time(0));
    torch::cuda::manual_seed_all(time(0));

    int imgWidth = 640;
    int imgHeight = 352;
    int batchSize = 36;
    int totalEpoch = 1000;
    int startEpoch = 1; //minimum 1
    bool specificCheckpointFlag = true;
    std::string specificCheckpointModelName;
    if(specificCheckpointFlag) {
        specificCheckpointModelName = "/home/chungbuk/dk/2023/train_bts/build/pre_20230315/01000+model.pt";
    }
    if(startEpoch < 1) {
        std::cerr << "check startEpoch!!!" << std::endl;
        exit(1);
    }
    int useThreads = 20; //for loading dataset
    int cudaId = 0;
    torch::Device device = torch::Device(cv::format("cuda:%d", cudaId));
    std::vector<std::string> imgNameFormatLet;
    std::vector<std::string> depthGtNameFormatLet;

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_depth/Own_S_depth/*/png/*/*.png");
    depthGtNameFormatLet.push_back("/home/chungbuk/dk/db_depth/Own_S_depth/*/depth_map/*/*.png");

    std::vector<std::pair<std::string, std::string>> dataPairLet;
    for(int j = 0; j < imgNameFormatLet.size(); j++) {
        std::string imgNameFormat = imgNameFormatLet[j];
        std::string depthGtNameFormat = depthGtNameFormatLet[j];
        std::vector<std::filesystem::path> imgFilesystemPathLet = glob::glob(imgNameFormat);
        std::vector<std::filesystem::path> depthGtFilesystemPathLet = glob::glob(depthGtNameFormat);
        std::sort(imgFilesystemPathLet.begin(), imgFilesystemPathLet.end());
        std::sort(depthGtFilesystemPathLet.begin(), depthGtFilesystemPathLet.end());
        if(imgFilesystemPathLet.size() != depthGtFilesystemPathLet.size()) {
            std::cerr << "check data pair at db" << j << "!!!" << std::endl;
            exit(1);
        }
        for(int i = 0; i < imgFilesystemPathLet.size(); i++) {
            dataPairLet.push_back(std::make_pair(imgFilesystemPathLet[i].string(), depthGtFilesystemPathLet[i].string()));
        }
    }
    std::srand(time(0));
    std::random_shuffle(dataPairLet.begin(), dataPairLet.end());
    auto dataset = CustomDatasetDepthEstimation(dataPairLet, imgWidth, imgHeight).map(torch::data::transforms::Stack<>());
    int dataSize = dataset.size().value();
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batchSize).workers(useThreads));
    BTS model("Own_S", 120.0,
              std::vector<int64_t>({96, 96, 192, 384, 2208}), 512,
              48, std::vector<int64_t>({6, 12, 36, 24}),
              96, 4, 0.0, 1000, false,
              std::vector<std::string>({"relu0", "pool0", "transition1", "transition2", "norm5"}));

    model->to(device);
    if(startEpoch == 1 && !specificCheckpointFlag) {
        model->apply(weights_init_xavier);
    }


    for(const auto &pair : model->named_parameters()) {
        std::string key = pair.key();
        float maxVal = torch::max(pair.value()).item<float>();
        float minVal = torch::min(pair.value()).item<float>();
        std::cout << key + " | max value: " << maxVal << ", min value: " << minVal << std::endl;
        // std::cout << key << std::endl;
    }

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(0.0001).eps(0.001));

    SilogLoss criterion(0.85);

    cv::Mat graph = cv::Mat(1000, totalEpoch, CV_8UC3, cv::Scalar(255, 255, 255));
    if(startEpoch != 1) {
        std::string startEpochStr = cv::format("%05d+", startEpoch - 1);
        graph = cv::imread(startEpochStr + "loss.png", cv::IMREAD_UNCHANGED);
        torch::load(model, startEpochStr + "model.pt");
        torch::load(optimizer, startEpochStr + "optimizer.pt");
    }
    if(specificCheckpointFlag) {
        torch::load(model, specificCheckpointModelName);
    }

    for(int i = startEpoch; i <= totalEpoch; i++) {
        model->train();
        float runningLoss = 0;
        int epoch = i;
        int forIdx = 0;
        std::chrono::system_clock::time_point currTick;
        std::chrono::system_clock::time_point prevTick = std::chrono::system_clock::now();
        for(const auto& batch : *dataLoader) {
            torch::Tensor imgBatchTensor = batch.data.to(device, true);
            torch::Tensor depthGtBatchTensor = batch.target.to(device, true);
            torch::Tensor depthPredBatchTensor = std::get<4>(model(imgBatchTensor, torch::Tensor()));
            torch::Tensor maskBatchTensor = depthGtBatchTensor > 1.0;
            maskBatchTensor = maskBatchTensor.to(torch::kBool).to(device, true);
            torch::Tensor lossTensor = criterion(depthPredBatchTensor, depthGtBatchTensor, maskBatchTensor);
            float localLoss = lossTensor.item<float>();
            runningLoss += localLoss; // runningLoss/dataIdx can be used <-> localLoss

            optimizer.zero_grad();
            lossTensor.backward();
            optimizer.step();
            int dataIdx = std::min(dataSize, (forIdx + 1)*batchSize);
            forIdx++;

            torch::Tensor rTensor = imgBatchTensor[0][0].mul(0.229).add(0.485).mul(255.0).to(torch::kUInt8).to(torch::kCPU);
            torch::Tensor gTensor = imgBatchTensor[0][1].mul(0.224).add(0.456).mul(255.0).to(torch::kUInt8).to(torch::kCPU);
            torch::Tensor bTensor = imgBatchTensor[0][2].mul(0.225).add(0.406).mul(255.0).to(torch::kUInt8).to(torch::kCPU);
            std::vector<cv::Mat> bgr(3);
            bgr[0] = cv::Mat(imgHeight, imgWidth, CV_8UC1, bTensor.data_ptr<uint8_t>());
            bgr[1] = cv::Mat(imgHeight, imgWidth, CV_8UC1, gTensor.data_ptr<uint8_t>());
            bgr[2] = cv::Mat(imgHeight, imgWidth, CV_8UC1, rTensor.data_ptr<uint8_t>());
            cv::Mat img;
            cv::merge(bgr, img);
            torch::Tensor depthPredImgTensor = depthPredBatchTensor[0].mul(100.0).to(torch::kInt16).to(torch::kCPU);
            cv::Mat depthPredImg = cv::Mat(imgHeight, imgWidth, CV_16UC1, (uint16_t*)depthPredImgTensor.data_ptr<int16_t>());

            cv::circle(graph, cv::Point(epoch, graph.rows - (int)(200*localLoss + 1)), 3, cv::Scalar(255, 0, 0), -1);

            currTick = std::chrono::system_clock::now();
            std::chrono::duration<double> processingTime = currTick - prevTick;
            prevTick = currTick;
            size_t cudaFreeByteT;
            size_t cudaTotalByteT;
            cudaSetDevice(cudaId);
            cudaMemGetInfo(&cudaFreeByteT, &cudaTotalByteT);
            float cudaFree = (float)(cudaFreeByteT/1024.0)/1024.0;
            float cudaTotal = (float)(cudaTotalByteT/1024.0)/1024.0;
            float cudaUsed = cudaTotal - cudaFree;
            std::cout << "Epoch " << epoch << ": " << dataIdx << "/" << dataSize << " | Loss: " << localLoss << " | Processing Time: " << processingTime.count() << "s" << " | CUDA Memory Usage: " << cudaUsed << "MB/" << cudaTotal << "MB" << std::endl;

            if((forIdx == std::ceil((float)dataSize/(float)(batchSize))) && (epoch%10 == 0)) {
                std::string epochStr = cv::format("%05d+", epoch);
                torch::save(model, epochStr + "model.pt");
                torch::save(optimizer, epochStr + "optimizer.pt");
                cv::imwrite(epochStr + "loss.png", graph);
                cv::imwrite(epochStr + "valid_img.png", img);
                cv::imwrite(epochStr + "valid_depth_pred_img.png", depthPredImg);
            }
        }
    }

    return 0;
}



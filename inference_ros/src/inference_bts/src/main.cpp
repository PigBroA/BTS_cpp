#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <bts/densenet.h>
#include <bts/bts.h>

int cudaId = 0;
torch::Device device = torch::Device(cv::format("cuda:%d", cudaId));

BTS model("Own_S", 120.0,
          std::vector<int64_t>({96, 96, 192, 384, 2208}), 512,
          48, std::vector<int64_t>({6, 12, 36, 24}),
          96, 4, 0.0, 1000, false,
          std::vector<std::string>({"relu0", "pool0", "transition1", "transition2", "norm5"}));

void callback(const sensor_msgs::CompressedImage::ConstPtr& msg);


int main(int argc, char** argv) {
    ros::init(argc, argv, "inference_bts");
    ros::NodeHandle nh("~");

    std::string imgTopic;
    std::string modelWeightName;
    nh.param<std::string>("imgTopic", imgTopic, "/front_60degree_camera/compressed");
    nh.param<std::string>("modelWeightName", modelWeightName, "/home/ldk/my_code/ros/test_ws/src/inference_bts/data/00020+model.pt");

    ros::Subscriber sub = nh.subscribe(imgTopic, 1, callback);

    model->to(device);
    torch::load(model, modelWeightName);

    cv::namedWindow("img", cv::WINDOW_NORMAL);

    ros::spin();
}

void callback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    std::chrono::system_clock::time_point startTick = std::chrono::system_clock::now();

    model->eval();
    torch::NoGradGuard noGrad;

    cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::resize(img, img, cv::Size(640, 352));

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    torch::Tensor r = torch::from_blob(channels[2].ptr(), {img.rows, img.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.485).div(0.229);
    torch::Tensor g = torch::from_blob(channels[1].ptr(), {img.rows, img.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.456).div(0.224);
    torch::Tensor b = torch::from_blob(channels[0].ptr(), {img.rows, img.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.406).div(0.225);
    torch::Tensor imgBatchTensor = torch::cat({r, g, b}).view({1, 3, img.rows, img.cols}).to(device);

    torch::Tensor depthPredBatchTensor = std::get<4>(model(imgBatchTensor, torch::Tensor()));
    torch::Tensor depthPredImgTensor = depthPredBatchTensor[0].mul(100.0).to(torch::kInt16).to(torch::kCPU);
    cv::Mat depthPredImg = cv::Mat(img.rows, img.cols, CV_16UC1, (uint16_t*)depthPredImgTensor.data_ptr<int16_t>());
    depthPredImg /= 100;
    depthPredImg.convertTo(depthPredImg, CV_8UC1);
    depthPredImg *= 2;
    depthPredImg *= (180.0 / 255.0);
    cv::Mat maxMap = cv::Mat(depthPredImg.rows, depthPredImg.cols, CV_8UC1, 255);
    std::vector<cv::Mat> channelValueLet;
    cv::Mat depthPredColorImg;
    channelValueLet.push_back(depthPredImg); // H
    channelValueLet.push_back(maxMap);   // S
    channelValueLet.push_back(maxMap);   // V
    cv::merge(channelValueLet, depthPredColorImg);
    cv::cvtColor(depthPredColorImg, depthPredColorImg, cv::COLOR_HSV2BGR);

    img += depthPredColorImg;

    cv::imshow("img", img);
    int wk = cv::waitKey(1);
    if(wk == 27) {
        exit(1);
    }

    std::chrono::duration<double> processingTime = std::chrono::system_clock::now() - startTick;
    // std::cout << processingTime.count() << std::endl;
}

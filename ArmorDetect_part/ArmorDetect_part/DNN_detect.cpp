#include "DNN_detect.h"

using namespace cv;

dnn::Net DNN_detect::read_net(const std::string& net_path) {
    return dnn::readNetFromONNX(net_path);//类 DNN_detect的 基于深度学习网络对数字图片进行识别
}
//对原始图片预处理，将其转化为神经网络可以处理的 blob 
Mat DNN_detect::img_processing(Mat ori_img, bool to_gray) {
    Mat out_blob;
    if (to_gray) {
        cvtColor(ori_img, ori_img, COLOR_BGR2GRAY);//灰度化
        //按照指定类型转换
        ori_img.convertTo(ori_img, CV_32FC1, 1.0f / 255.0f);//像素值除255--放缩到[0,1] 便于计算 -- 避免溢出或精度不足
    }
    else { ori_img.convertTo(ori_img, CV_32FC3, 1.0f / 255.0f); }//同上 (三通道)
    //1.输入图像矩阵 2.输出四维张量 3.放缩 4.SIZE 5.mean/swapRB crop ddepth
    dnn::blobFromImage(ori_img, out_blob, 1.0, Size(IMG_SIZE, IMG_SIZE));//得到blob
    return out_blob;
}

void DNN_detect::net_forward(const Mat& blob, dnn::Net net, int& id, double& confidence) {
    net.setInput(blob);
    Mat outputs = net.forward();//forward() 获取网络输出
    float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());//找范围内最大元素赋值给max_prob
    cv::Mat softmax_prob;
    //softmax归一化
    cv::exp(outputs - max_prob, softmax_prob);      
    //元素求和--得概率分布
    float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
    softmax_prob /= sum;
    cv::Point class_id;
    //(chapter.直方图)
    minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id);//找矩阵最大最小值 -- 返回位置及数值
    id = class_id.x + 1;
}

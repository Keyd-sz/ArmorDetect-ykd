#include "DNN_detect.h"

using namespace cv;

dnn::Net DNN_detect::read_net(const std::string& net_path) {
    return dnn::readNetFromONNX(net_path);//�� DNN_detect�� �������ѧϰ���������ͼƬ����ʶ��
}
//��ԭʼͼƬԤ��������ת��Ϊ��������Դ���� blob 
Mat DNN_detect::img_processing(Mat ori_img, bool to_gray) {
    Mat out_blob;
    if (to_gray) {
        cvtColor(ori_img, ori_img, COLOR_BGR2GRAY);//�ҶȻ�
        //����ָ������ת��
        ori_img.convertTo(ori_img, CV_32FC1, 1.0f / 255.0f);//����ֵ��255--������[0,1] ���ڼ��� -- ��������򾫶Ȳ���
    }
    else { ori_img.convertTo(ori_img, CV_32FC3, 1.0f / 255.0f); }//ͬ�� (��ͨ��)
    //1.����ͼ����� 2.�����ά���� 3.���� 4.SIZE 5.mean/swapRB crop ddepth
    dnn::blobFromImage(ori_img, out_blob, 1.0, Size(IMG_SIZE, IMG_SIZE));//�õ�blob
    return out_blob;
}

void DNN_detect::net_forward(const Mat& blob, dnn::Net net, int& id, double& confidence) {
    net.setInput(blob);
    Mat outputs = net.forward();//forward() ��ȡ�������
    float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());//�ҷ�Χ�����Ԫ�ظ�ֵ��max_prob
    cv::Mat softmax_prob;
    //softmax��һ��
    cv::exp(outputs - max_prob, softmax_prob);      
    //Ԫ�����--�ø��ʷֲ�
    float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
    softmax_prob /= sum;
    cv::Point class_id;
    //(chapter.ֱ��ͼ)
    minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id);//�Ҿ��������Сֵ -- ����λ�ü���ֵ
    id = class_id.x + 1;
}

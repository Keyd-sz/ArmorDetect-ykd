#ifndef NUM_DECTEC_ARMORDETECTOR_H
#define NUM_DECTEC_ARMORDETECTOR_H

#pragma once//��ֹ�ظ�����

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include "DNN_detect.h"
#include "SpinTracker.h"


using namespace std;

#define POINT_DIST(p1,p2) std::sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y))
struct SRC {
public:
    Mat img;
    int time_stamp; //ʱ���--�������ֲ�ͬ֡
};

//�����ṹ��
struct Light : public cv::RotatedRect     //�����ṹ��
{
    Light();
    explicit Light(cv::RotatedRect& box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) {return a.y < b.y; });//
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;
        height = POINT_DIST(top, bottom);
        width = POINT_DIST(p[0], p[1]);
        angle = (top.x <= bottom.x) ? (box.angle) : (90 + box.angle);//����angle

    }
    int lightColor;
    cv::Point2f top;
    cv::Point2f bottom;
    double angle;
    double height;
    double width;
};


//����
class ArmorDetector :public robot_state  //robot_state-->
{
public:
    ArmorDetector(); //���캯����ʼ��

    Armor autoAim(const Mat& src, int timestamp); //������Ŀ�������ת��������ͷԭ��С��



private:
    int lostCnt;//
    int binThresh;//��������ֵ

    //light_judge_condition
    double light_max_angle;
    double light_min_hw_ratio;
    double light_max_hw_ratio;   // different distance and focus
    double light_min_area_ratio;   // RotatedRect / Rect
    double light_max_area_ratio;


    //armor_judge_condition
    double armor_max_wh_ratio;
    double armor_min_wh_ratio;
    double armor_max_angle;
    double armor_height_offset;
    double armor_ij_min_ratio;
    double armor_ij_max_ratio;

    //armor_grade_condition
    double big_wh_standard;
    double small_wh_standard;
    double near_standard;
    int grade_standard;

    //armor_grade_project_ratio
    double id_grade_ratio;
    double wh_grade_ratio;
    double height_grade_ratio;
    double near_grade_ratio;
    double angle_grade_ratio;

    bool Lost;
    bool smallArmor;

    cv::Mat _src;  // �ü�src���ROI--region of insterest
    cv::Mat _binary;
    std::vector<cv::Mat> temps;

    cv::Rect detectRoi;  //Ϊ�˰�srcͨ��roi���_src

    Armor lastArmor;

    std::vector<Light> candidateLights; // ɸѡ�ĵ���
    std::vector<Armor> candidateArmors; // ɸѡ��װ�װ�
    Armor finalArmor;  // ����װ�װ�

    //STL�������� �ṩһ��һ��hash
    std::map<int, int> new_armors_cnt_map;          //װ�װ����map����¼����װ�װ���
    std::multimap<int, SpinTracker> trackers_map;  //����mapһ������"tracker_mapԤ����"
    const int max_delta_t = 50;                //ʹ��ͬһԤ���������ʱ����(ms)
    const double max_delta_dist = 40;              // ���׷�پ���
    std::map<int, SpinHeading> spin_status_map;     // ��¼�ó�С����״̬state��δ֪��˳ʱ�룬��ʱ�룩
    //std::map<int, double> spin_score_map;           // ��¼��װ�װ�С���ݿ����Է���������0Ϊ��ʱ����ת��С��0Ϊ˳ʱ����ת
    std::unordered_map<int, double> spin_score_map;           // ��ϣ��ʵ�ֵ������������ ����ʱ�临�Ӷ�

    double anti_spin_max_r_multiple = 4.5;         // �������������������ݷ������ӱ���
    int anti_spin_judge_low_thres = 2e3;           // С�ڸ���ֵ��Ϊ�ó��ѹر�����
    int anti_spin_judge_high_thres = 2e4;          // ���ڸ���ֵ��Ϊ�ó��ѿ�������


    bool updateSpinScore();

    cv::Point2f dst_p[4] = { cv::Point2f(0,60),cv::Point2f(0,0),cv::Point2f(30,0),cv::Point2f(30,60) };

    void setImage(const cv::Mat& src); //��ͼ���������

    void findLights(); //�ҵ�����ȡ��ѡƥ��ĵ���

    void matchLights(); //ƥ�������ȡ��ѡװ�װ�


    void chooseTarget(int time_stamp); //�ҳ����ȼ���ߵ�װ�װ�

    bool isLight(Light& light, std::vector<cv::Point>& cnt);//����light����Ϣ �� ��contours�õ������

    bool conTain(cv::RotatedRect& match_rect, std::vector<Light>& Lights, size_t& i, size_t& j);

    int armorGrade(const Armor& checkArmor);

    void detectNum(Armor& armor);

    //����inline ***
    static inline void dnn_detect(cv::Mat frame, Armor& armor)// ���øú������ɷ�������ID
    {
        return DNN_detect::net_forward(DNN_detect::img_processing(std::move(frame), TO_GRAY), DNN_detect::read_net(NET_PATH), armor.id, armor.confidence);
    }

    static inline bool makeRectSafe(cv::Rect& rect, cv::Size size) {
        if (rect.x < 0)
            rect.x = 0;
        if (rect.x + rect.width > size.width)
            rect.width = size.width - rect.x;
        if (rect.y < 0)
            rect.y = 0;
        if (rect.y + rect.height > size.height)
            rect.height = size.height - rect.y;
        if (rect.width <= 0 || rect.height <= 0)
            // ������־����ǿյģ��򷵻�false
            return false;
        return true;
    }

    static inline bool height_sort(Armor& candidate1, Armor& candidate2)
    {
        return candidate1.size.height > candidate2.size.height;
    }
};



#endif //NUM_DECTEC_ARMORDETECTOR_H
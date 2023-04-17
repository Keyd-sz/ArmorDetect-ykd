
#include <opencv2/opencv.hpp>
#include "robot_state.h"
#ifndef SHSIHI_ARMORTRACKER_H
#define SHSIHI_ARMORTRACKER_H

using namespace cv;
using namespace std;



//װ�װ�ṹ��
struct Armor : public cv::RotatedRect    //װ�װ�ṹ��
{
    Armor() //��ʼ��
    {
        light_height_rate = 0;
        confidence = 0;
        id = 0;
        type = SMALL;
    }
    explicit Armor(cv::RotatedRect& box) : cv::RotatedRect(box)
    {
        light_height_rate = 0;
        confidence = 0;
        id = 0;
        type = SMALL;
    }
    double light_height_rate;  // ���ҵ����߶ȱ�
    double confidence;
    int id;  // װ�װ����
    EnermyType type;  // װ�װ�����
    int hit_score; // �������
    //    int area;  // װ�װ����
};


class SpinTracker
{
public:
    Armor last_armor;                       //����װ�װ�
    bool is_initialized;                    //�Ƿ���ɳ�ʼ��
    int last_timestamp;                     //����װ�װ�ʱ���
    const int max_history_len = 4;          //��ʷ��Ϣ������󳤶�

    std::deque<Armor> history_info;  // Ŀ����� ��Armor

    explicit SpinTracker(const Armor& src, int src_timestamp);
    bool update_tracker(const Armor& new_armor, int timestamp);
};
#endif //SHSIHI_ARMORTRACKER_H

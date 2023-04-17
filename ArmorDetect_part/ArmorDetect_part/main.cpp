#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ArmorDetector.hpp"

using namespace cv;
using namespace std;

int main()
{
    ArmorDetector autoShoot;
    Armor autoTarget;

    // �򿪱�������ͷ
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Failed to open camera!" << endl;
        return -1;
    }

    while (true)
    {
        Mat src;
        auto time_start = chrono::steady_clock::now();

        // ������ͷ��ȡһ֡ͼ��
        cap.read(src);

        if (!src.empty())
        {
            auto time_cap = chrono::steady_clock::now();
            int time_stamp = (int)(chrono::duration<double, std::milli>(time_cap - time_start).count());

            // ����װ�װ��⣬���ؼ����
            autoTarget = autoShoot.autoAim(src, time_stamp);
            imshow("src", src);
            if (!autoTarget.size.empty())
            {
                cout << "get target" << endl;
            }
        }

        // �ȴ���������һ��ʱ����˳�ѭ��
        if (waitKey(1) == 27) // ESC ��
        {
            break;
        }
    }

    return 0;
}
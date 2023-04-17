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

    // 打开本地摄像头
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

        // 从摄像头读取一帧图像
        cap.read(src);

        if (!src.empty())
        {
            auto time_cap = chrono::steady_clock::now();
            int time_stamp = (int)(chrono::duration<double, std::milli>(time_cap - time_start).count());

            // 进行装甲板检测，返回检测结果
            autoTarget = autoShoot.autoAim(src, time_stamp);
            imshow("src", src);
            if (!autoTarget.size.empty())
            {
                cout << "get target" << endl;
            }
        }

        // 等待按键或者一段时间后退出循环
        if (waitKey(1) == 27) // ESC 键
        {
            break;
        }
    }

    return 0;
}
#include "ArmorDetector.hpp"

#define BINARY_SHOW

//#define DRAW_LIGHTS_CONTOURS
//#define DRAW_LIGHTS_RRT

//#define DRAW_ARMORS_RRT
//#define DRAW_FINAL_ARMOR_CLASS
#define DRAW_FINAL_ARMOR_MAIN

using namespace cv;
using namespace std;

//���캯�����Ը������Գ�ʼ����hpp����˵���������Ե�����
ArmorDetector::ArmorDetector()
{
    lastArmor = Armor();
    detectRoi = cv::Rect();
    smallArmor = false;
    lostCnt = 0;
    Lost = true;



    //binary_thresh������������������������������������������Ҫ�ģ�������
    binThresh = 100;   // blue 100  red  70

    //light_judge_condition
    light_max_angle = 45.0;
    light_min_hw_ratio = 1;
    light_max_hw_ratio = 15;   // different distance and focus
    light_min_area_ratio = 0.6;   // RotatedRect / Rect
    light_max_area_ratio = 1.0;

    //armor_judge_condition
    armor_max_wh_ratio = 3;
    armor_min_wh_ratio = 1.0;
    armor_max_angle = 20.0;
    armor_height_offset = 0.25;
    armor_ij_min_ratio = 0.5;
    armor_ij_max_ratio = 2.0;

    //armor_grade_condition
    big_wh_standard = 3.5; // 4������ѣ�3.5���ϱȽ������������޸�
    small_wh_standard = 2; // 2.5������ѣ�2���ϱȽ������������޸�
    near_standard = 500;

    //armor_grade_project_ratio
    id_grade_ratio = 0.2;
    wh_grade_ratio = 0.3;
    height_grade_ratio = 0.2;
    near_grade_ratio = 0.2;
    angle_grade_ratio = 0.1;

    grade_standard = 70; // �����
}

void ArmorDetector::setImage(const Mat& src)
{
    //��ȡ��һ�μ�⵽��װ�װ����ĵ�λ��
    const Point& lastCenter = lastArmor.center;
    if (lastCenter.x == 0 || lastCenter.y == 0)//��һ��δ��⵽װ�װ�
    {
        _src = src;//����ͼ����вü�
        detectRoi = Rect(0, 0, src.cols, src.rows);//ԭʼͼ��Ϊ�������
    }
    else //��һ�μ�⵽װ�װ�
    {
        //Rect rect = finalRect;
        Rect rect = lastArmor.boundingRect();

        double scale_w = 2; //�������������ű���
        double scale_h = 2; //�������߶����ű���
        int lu_x_offset = 0;//���Ͻ�ƫ����
        int rd_x_offset = 0;//���½�ƫ����
        // ��ȡƫ����
        if (lastArmor.light_height_rate > 1) //��һ�μ�⵽��Ϊ��������
            lu_x_offset = 6 * (pow(lastArmor.light_height_rate - 1, 0.6) + 0.09) * rect.width; //���Ͻ�ƫ����
        else //��һ�μ�⵽��Ϊ��������
            rd_x_offset = 6 * (pow(1 - lastArmor.light_height_rate, 0.6) + 0.15) * rect.width; //���½�ƫ����

        //��ȡ��ǰ֡��roi
        int w = int(rect.width * scale_w);
        int h = int(rect.height * scale_h);
        int x = max(lastCenter.x - w - lu_x_offset, 0);
        int y = max(lastCenter.y - h, 0);
        Point luPoint = Point(x, y);//���Ͻ�����
        x = std::min(lastCenter.x + w + rd_x_offset, src.cols);
        y = std::min(lastCenter.y + h, src.rows);
        Point rdPoint = Point(x, y);//���½�����
        detectRoi = Rect(luPoint, rdPoint);

        //���������򳬳�ͼ��߽磬������ԭʼͼ�񲢽������������Ϊ����ͼ��
        if (!makeRectSafe(detectRoi, src.size())) {
            lastArmor = Armor();
            detectRoi = Rect(0, 0, src.cols, src.rows);
            _src = src;
        }
        else
            src(detectRoi).copyTo(_src);//�ü�ͼ��
    }
    //��ֵ��
    Mat gray;
    cvtColor(_src, gray, COLOR_BGR2GRAY);
    threshold(gray, _binary, binThresh, 255, THRESH_BINARY);//����������������������������������
#ifdef BINARY_SHOW
    imshow("_binary", _binary);
#endif //BINARY_SHOW
}

//�Ƿ��ǵ���
bool ArmorDetector::isLight(Light& light, vector<Point>& cnt)
{

    double height = light.height;
    double width = light.width;

    if (height <= 0 || width <= 0)
        return false;

    //��һ��Ҫ���ڿ�
    bool standing_ok = height > width;

    //�߿�������
    double hw_ratio = height / width;
    bool hw_ratio_ok = light_min_hw_ratio < hw_ratio&& hw_ratio < light_max_hw_ratio;

    double area_ratio = contourArea(cnt) / (height * width);
    bool area_ratio_ok = light_min_area_ratio < area_ratio&& area_ratio < light_max_area_ratio;


    //�����Ƕ�����
    bool angle_ok = fabs(90.0 - light.angle) < light_max_angle || light.angle == 0;

    //�����жϵ������ܼ�
    bool is_light = hw_ratio_ok && area_ratio_ok && angle_ok && standing_ok;


    //    if(!is_light)
    //    {
    //        cout<<hw_ratio<<"    "<<area_ratio<<"    "<<light.angle<<endl;
    //    }


    return is_light;
}

void ArmorDetector::findLights()
{
    //������
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(_binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

#ifdef DRAW_LIGHTS_CONTOURS
    //������
    for (int i = 0; i < contours.size(); i++)
        cv::drawContours(_src, contours, i, Scalar(255, 0, 0), 2, LINE_8);
    imshow("contours_src", _src);
#endif
    //�����⵽��������������2������Ϊδ��⵽��������պ�ѡ�������ϲ��˳�����
    if (contours.size() < 2)
    {
        printf("no 2 contours\n");
        candidateLights.clear();
        return;
    }
    //��������
    for (auto& contour : contours)
    {
        RotatedRect r_rect = minAreaRect(contour); //��ȡ��С��Ӿ���
        Light light = Light(r_rect); //������С��Ӿ��ι�������

        if (isLight(light, contour)) //�жϵ����Ƿ����Ҫ��
        {
            //��Ӿ���תΪ���������ڱ�����ɫ
            cv::Rect rect = r_rect.boundingRect();

            if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= _src.cols &&
                0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= _src.rows)
            {
                int sum_r = 0, sum_b = 0;
                cv::Mat roi = _src(rect);
                // ����ROI
                for (int i = 0; i < roi.rows; i++)
                {
                    for (int j = 0; j < roi.cols; j++)
                    {
                        if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) // ֻ���������е�����������
                        {
                            sum_r += roi.at<cv::Vec3b>(i, j)[2];
                            sum_b += roi.at<cv::Vec3b>(i, j)[0];
                        }
                    }
                }
                //cout<<sum_r<<"           "<<sum_b<<endl;
                // Sum of red pixels > sum of blue pixels ?
                light.lightColor = sum_r > sum_b ? RED : BLUE;

                // ��ɫ�����ϵ�ط��ľͲ�����
                if (light.lightColor == 1)
                {
                    //������Ҫ��ĵ������ӵ���ѡ����������
                    candidateLights.emplace_back(light);
#ifdef DRAW_LIGHTS_RRT
                    Point2f vertice_lights[4];
                    light.points(vertice_lights);
                    for (int i = 0; i < 4; i++) {
                        line(_src, vertice_lights[i], vertice_lights[(i + 1) % 4], CV_RGB(255, 0, 0), 2, LINE_8);
                    }
                    //circle(_src,light.center,5,Scalar(0,0,0),-1);
                    imshow("lights-show-_src", _src);
#endif //DRAW_LIGHTS_RRT
                }
            }
        }
    }
    //cout<<"dengtiao  geshu:  "<<candidateLights.size()<<endl;
    //�����ѡ���������е�����������2�����˳�����
    if (candidateLights.size() < 2)
    {
        return;
    }
}

//ƥ�����
void ArmorDetector::matchLights()
{
    //���������������
    if (candidateLights.size() < 2)
    {
        printf("no 2 lights\n");
        candidateLights.clear();//��ѡ�������
        return;
    }

    // ����ת���δ���������
    sort(candidateLights.begin(), candidateLights.end(),
        [](RotatedRect& a1, RotatedRect& a2) {
            return a1.center.x < a2.center.x; });
    //    cout << "candidateLights size: " << candidateLights.size() << endl;
    //��������I
    for (size_t i = 0; i < candidateLights.size() - 1; i++)
    {
        Light lightI = candidateLights[i];
        Point2f centerI = lightI.center;
        //��������J
        for (size_t j = i + 1; j < candidateLights.size(); j++)
        {

            Light lightJ = candidateLights[j];
            Point2f centerJ = lightJ.center;
            double armorWidth = POINT_DIST(centerI, centerJ) - (lightI.width + lightJ.width) / 2.0; // ��װ�װ����
            double armorHeight = (lightI.height + lightJ.height) / 2.0; // ��װ�װ�߶�
            double armor_ij_ratio = lightI.height / lightJ.height; // �����ҵ����߶ȱ�

            //���߱�ɸѡ����
            bool whratio_ok = armor_min_wh_ratio < armorWidth / armorHeight && armorWidth / armorHeight < armor_max_wh_ratio;

            //�Ƕ�ɸѡ������Ϊɶ�����ˡ�
//            bool angle_ok = fabs(lightI.angle - lightJ.angle) < armor_max_angle;

            //�������������ĵ�߶Ȳ�ɸѡ����
            bool height_offset_ok = fabs(lightI.height - lightJ.height) / armorHeight < armor_height_offset;

            //���ҵ����ĸ߶ȱ�
            bool ij_ratio_ok = armor_ij_min_ratio < armor_ij_ratio&& armor_ij_ratio < armor_ij_max_ratio;

            //��������
            bool is_like_Armor = whratio_ok && height_offset_ok && ij_ratio_ok;
            //            cout << "-----------------------------------------------------------------" << endl;
            //            cout << "is_like_Armor: " << is_like_Armor << endl;
            //            cout << "LightI_angle :   "<<lightI.angle<<"   LightJ_angle :   "<<lightJ.angle<<"     "<<fabs(lightI.angle - lightJ.angle)<<endl;
            //            cout << "    w/h      :   "<<armorWidth/armorHeight<<endl;
            //            cout << "height-offset:   "<<fabs(lightI.height - lightJ.height) / armorHeight<<endl;
            //            cout << " height-ratio:   "<<armor_ij_ratio<<endl;
            //            cout << "-----------------------------------------------------------------" << endl;
            if (is_like_Armor)//��������
            {
                //��ʼ��װ�װ�����
                Point2f armorCenter = (centerI + centerJ) / 2.0;
                double armorAngle = atan2((centerI.y - centerJ.y), fabs(centerI.x - centerJ.x));
                RotatedRect armor_rrect = RotatedRect(armorCenter,
                    Size2f(armorWidth, armorHeight),
                    -armorAngle * 180 / CV_PI);
                //                cout << "is_like_armor" << endl;
                //����ж�һ�Σ�������ô�ж���conTain�
                if (!conTain(armor_rrect, candidateLights, i, j))
                {
                    candidateArmors.emplace_back(armor_rrect);//�����ѡװ�װ�
                    //��ӡ��������
                    //#ifdef DRAW_ARMORS_RRT
                    ////                    cout<<"LightI_angle :   "<<lightI.angle<<"   LightJ_angle :   "<<lightJ.angle<<"     "<<fabs(lightI.angle - lightJ.angle)<<endl;
                    ////                    cout<<"armorAngle   :   "<<armorAngle * 180 / CV_PI <<endl;
                    ////                    cout<<"    w/h      :   "<<armorWidth/armorHeight<<endl;
                    ////                    cout<<"height-offset:   "<<fabs(lightI.height - lightJ.height) / armorHeight<<endl;
                    ////                    cout<<" height-ratio:   "<<armor_ij_ratio<<endl;
                    //
                    //                    Point2f vertice_armors[4];
                    //                    armor_rrect.points(vertice_armors);
                    //                    for (int i = 0; i < 4; i++) {
                    //                        line(_src, vertice_armors[i], vertice_armors[(i + 1) % 4], CV_RGB(0, 255, 0),2,LINE_8);
                    //                    }
                    //                    circle(_src,armorCenter,15,Scalar(0,255,255),-1);
                    //                    imshow("armors-show-_src", _src);
                    //#endif //DRAW_ARMORS_RRT
                    (candidateArmors.end() - 1)->light_height_rate = armor_ij_ratio;// ��װ�װ���Ϣ�м�������߶ȱ���Ϣ
                }
            }

        }
    }

    // ���û���ҵ���ѡװ�װ壬ֱ�ӷ���
    if (candidateArmors.empty())
    {
        //        cout << "return" << endl;
        return;
    }

}

//ѡ��Ŀ��
void ArmorDetector::chooseTarget(int timestamp)
{
    //��û�к�ѡװ�װ�
    if (candidateArmors.empty())
    {
        cout << "no target!!" << endl;
        finalArmor = Armor();
        lostCnt += 1;
    }
    else
    {

        cout << "get " << candidateArmors.size() << " target!!" << endl;

        // ��Ѻ�ѡװ�װ��±�
        int best_index = -1;
        int best_record = 0;//װ�װ�÷֣����ڱȽ��ĸ�װ�װ�����

        // ����װ�װ�߶��ɴ�С��������
        sort(candidateArmors.begin(), candidateArmors.end(), height_sort);

        // ����´������������ӳ�䡾���������������������������������ʲô������˽���map��
        new_armors_cnt_map.clear();

        for (int i = 0; i < candidateArmors.size(); ++i) {
            // ��ȡÿ����ѡװ�װ��id��type��������detectNum��������������������������������
            detectNum(candidateArmors[i]);
            //            candidateArmors[i].id = 4;
            //            candidateArmors[i].confidence = 99;

            int armor_id = candidateArmors[i].id;
            // ������Ŷ�С����ֵ�������Ǹ�װ�װ塾��������������������������һ�����Ŷ������︳ֵ��
            if (candidateArmors[i].confidence < 0.95) {
                continue;
            }
            // �����û��װ�װ�����򣬱��ΪSMALL��������һ����������������������������û��0��2��װ�װ�id�𣿣�
            if (armor_id == 0 || armor_id == 2) {
                candidateArmors[i].type = SMALL;
                continue;
            }

            // ��ʱֻ��������
            //�ж�װ�װ����С
            if (armor_id == 1)
                candidateArmors[i].type = BIG;
            else if (armor_id == 3 || armor_id == 4)
                candidateArmors[i].type = SMALL;


            auto predictors_with_same_key = trackers_map.count(armor_id);  // �Ѵ���id��Ԥ����������trackers_map��multimap��Ϊʲô����ֻ�ж�01������������

            if (predictors_with_same_key == 0)// �����װ�װ����͵�Ԥ����������
            {
                SpinTracker tracker(candidateArmors[i], timestamp);// �����µ�Ԥ����
                auto target_predictor = trackers_map.insert(make_pair(armor_id, tracker));// ��Ԥ��������map��target_predictor�ƺ�û�õ�������
                new_armors_cnt_map[armor_id]++;  // �����ʹ�������id������1��
            }
            else if (predictors_with_same_key == 1)// �����װ�װ����͵�Ԥ�����Ѵ���
            {
                auto candidate = trackers_map.find(armor_id);  // ԭ�е�ͬIDװ�װ壨candidate�ǵ�������
                auto delta_dist = POINT_DIST(candidateArmors[i].center, (*candidate).second.last_armor.center);  // ����װ�װ��ࡾsecondӦ��ָ����pair�ĵڶ��Ҳ����SpinTracker tracker��

                //��ƥ����ʹ�ô�ArmorTracker  �������С����ֵ&&���װ�װ��Rect�����ϸ�װ�װ�����ģ���ƥ��ɹ�
                if (delta_dist <= max_delta_dist && (*candidate).second.last_armor.boundingRect().contains(candidateArmors[i].center))
                {
                    (*candidate).second.update_tracker(candidateArmors[i], timestamp);  // ����װ�װ壨������װ�װ��Ŀ����и��£�����Ϊȷ����������װ�װ���ͬһ����
                    //                    cout << "updata tracker" << endl;
                }
                //����ƥ���򴴽���ArmorTracker
                else
                {
                    //��predictors_with_same_key == 0��������
                    SpinTracker tracker(candidateArmors[i], timestamp);
                    trackers_map.insert(make_pair(armor_id, tracker));
                    new_armors_cnt_map[armor_id]++;  // ͬ���Ͳ�ͬλ�ô���
                    //                    cout << "create tracker" << endl;

                }
            }


            //�����ɸѡװ�װ����ȼ�
            /*������ȼ�����ʶ��Ӣ��1��װ�װ壬���3��4�ţ������ֵĻ�1��100��3��4��80������������
             *1�����߱ȣ�ɸѡ����Ͳ���װ�װ壬����������װ�װ壩
             *2��װ�װ忿��ͼ������
             *3��װ�װ���б�Ƕ���С
             *4��װ�װ�����
             */
             //1�����߱���һ����׼ֵ�͵�ǰֵ����ֵ�����ڱ�׼ֵĬ����Ϊ1�����ϱ�׼������Ϊ�÷�
             //2������Сroi�ھ͸��֣����ڲ����֣�����ռ�Ƚϵͣ�
             //3��90�ȼ�ȥװ�װ�ĽǶȳ���90�õ���ֵ���ϱ�׼����Ϊ�÷�
             //4����ǰ�������֮ǰ��װ�װ���и��ɴ�С���򣬻�ȡ�����СֵȻ���һ�����ù�һ���ĸ߶�ֵ���ϱ�׼����Ϊ�÷�

            // �Ժ�ѡװ�װ�������֣�ѡ��������ߵ�װ�װ���Ϊ����Ŀ��
            int final_record = armorGrade(candidateArmors[i]);
            //            cout << "final_record: " << final_record << endl;
            // �Ժ�ѡװ�װ�������֣�ѡ��������ߵ�װ�װ���Ϊ����Ŀ��
            if (final_record > best_record && final_record > grade_standard)
            {
                best_record = final_record;
                best_index = i;//���װ�װ��±�
                //                cout << "best_index: " << best_index <<endl;
            }
        }


        //        cout << "tracker size: " << trackers_map.size() << endl;
        // ɾ��Ԥ����Map�й��õ�װ�װ�
        if (!trackers_map.empty())
        {
            //ά��Ԥ����Map��ɾ������֮ǰ��װ�װ�
            for (auto iter = trackers_map.begin(); iter != trackers_map.end();)
            {
                //ɾ��Ԫ�غ��������ʧЧ�������л�ȡ��һԪ��
                auto next = iter;
                // cout<<(*iter).second.last_timestamp<<"  "<<src.timestamp<<endl;
                if ((timestamp - (*iter).second.last_timestamp) > max_delta_t)
                    next = trackers_map.erase(iter);
                else
                    ++next;
                iter = next;
            }
        }

        //��������������������������������������������������������������������������������
        for (const auto& cnt : new_armors_cnt_map)
        {
            // ����װ�װ�����Ϊ1ʱ�������ݷ���
            if (cnt.second == 1)
            {
                auto same_armors_cnt = trackers_map.count(cnt.first);  // ��ͬ��װ�װ�����
                if (same_armors_cnt == 2)
                {
                    cout << "spin detect" << endl;
                    //��������ͬKeyԤ������ȷ�����Ҳ��Tracker
                    SpinTracker* new_tracker = nullptr;
                    SpinTracker* last_tracker = nullptr;
                    double last_armor_center;
                    double last_armor_timestamp;
                    double new_armor_center;
                    double new_armor_timestamp;
                    int best_prev_timestamp = 0;    //��ѡArmorTracker�����ʱ���
                    auto candiadates = trackers_map.equal_range(cnt.first);  // ��ȡͬ���͵�װ�װ�
                    for (auto iter = candiadates.first; iter != candiadates.second; ++iter)
                    {
                        //��δ��ɳ�ʼ������Ϊ����tracker
                        if (!(*iter).second.is_initialized && (*iter).second.last_timestamp == timestamp)
                        {
                            new_tracker = &(*iter).second;
                        }
                        else if ((*iter).second.last_timestamp > best_prev_timestamp && (*iter).second.is_initialized)
                        {
                            best_prev_timestamp = (*iter).second.last_timestamp;
                            last_tracker = &(*iter).second;
                        }

                    }
                    if (new_tracker != nullptr && last_tracker != nullptr)
                    {
                        new_armor_center = new_tracker->last_armor.center.x;
                        new_armor_timestamp = new_tracker->last_timestamp;
                        last_armor_center = last_tracker->last_armor.center.x;
                        last_armor_timestamp = last_tracker->last_timestamp;
                        auto spin_movement = new_armor_center - last_armor_center;  // ����x���꣺ �� - ��

                        cout << cnt.first << " : " << spin_movement << endl;
                        if (abs(spin_movement) > 10 && new_armor_timestamp == timestamp && last_armor_timestamp == timestamp)
                        {

                            // ���޸�Ԫ���������Ԫ��
                            if (spin_score_map.count(cnt.first) == 0)
                            {
                                spin_score_map[cnt.first] = 1000 * spin_movement / abs(spin_movement);
                            }
                            //  �����и�Ԫ����Ŀǰ��ת�������¼��ͬ,���Ŀǰ�������м���ͷ�
                            else if (spin_movement * spin_score_map[cnt.first] < 0)
                            {
                                spin_score_map[cnt.first] *= 0.5;
                            }
                            // �����и�Ԫ�������Ԫ��
                            else
                            {
                                spin_score_map[cnt.first] = anti_spin_max_r_multiple * spin_score_map[cnt.first];
                            }
                        }
                    }
                }
            }
        }

        updateSpinScore();

        bool is_target_spinning;
        std::vector<Armor> final_armors;
        SpinHeading spin_status;

        if (spin_status_map.count(candidateArmors[best_index].id) == 0)
        {
            spin_status = UNKNOWN;
            is_target_spinning = false;
        }
        else
        {
            spin_status = spin_status_map[candidateArmors[best_index].id];
            if (spin_status != UNKNOWN)
                is_target_spinning = true;
            else
                is_target_spinning = false;
        }
        //        cout << "is target spinning: " << is_target_spinning << endl;
        if (best_index != -1) {
            //            cout << "ID: " << candidateArmors[best_index].id << endl;
            //            cout << "Confidence: " << candidateArmors[best_index].confidence << endl;
            //            cout << "type: " << candidateArmors[best_index].type << endl;
            if (is_target_spinning) {
                auto ID_candiadates = trackers_map.equal_range(candidateArmors[best_index].id);

                for (auto iter = ID_candiadates.first; iter != ID_candiadates.second; ++iter) {
                    if ((*iter).second.last_timestamp == timestamp)  // ͬһ֡ͼ���װ�װ�
                    {
                        final_armors.push_back((*iter).second.last_armor);
                    }
                    else {
                        continue;
                    }
                }

                //������һ��װ�װ�
                if (final_armors.size() == 1) {
                    finalArmor = final_armors.at(0);
                }
                // ����������װ�װ�
                else if (final_armors.size() == 2) {
                    // ������װ�װ��������ѡȡ����ת������ͬ��װ�װ���и���
                    sort(final_armors.begin(), final_armors.end(),
                        [](Armor& prev, Armor& next) { return prev.center.x < next.center.x; });
                    // ��˳ʱ����תѡȡ�Ҳ�װ�װ����
                    if (spin_status == CLOCKWISE)
                        finalArmor = final_armors.at(1);
                    // ����ʱ����תѡȡ���װ�װ����
                    else if (spin_status == COUNTER_CLOCKWISE)
                        finalArmor = final_armors.at(0);
                }
            }
            else {
                finalArmor = candidateArmors[best_index];
            }
        }
    }

#ifdef DRAW_FINAL_ARMOR_CLASS
    Mat final_armor = _src.clone();
    Point2f vertice_armor[4];
    finalArmor.points(vertice_armor);
    for (int i = 0; i < 4; i++) {
        line(final_armor, vertice_armor[i], vertice_armor[(i + 1) % 4], CV_RGB(0, 255, 0));
    }
    imshow("final_armor-show", final_armor);
#endif //DRAW_FINAL_ARMOR_CLASS
}//������������������������������������������������������������

// ���麯��������ԭͼ��ʱ���
Armor ArmorDetector::autoAim(const Mat& src, int timestamp)
{
    // ��պ�ѡװ�װ�ͺ�ѡ����
    candidateArmors.clear();
    candidateLights.clear();

    // ��ʼ������װ�װ�Ϊ��
    finalArmor = Armor();

    // ��ͼ������Ϊ��ǰ֡�����ҵ�����
    setImage(src);
    findLights();

    // ƥ������γɺ�ѡװ�װ�
    matchLights();

    // ѡ��Ŀ��װ�װ�
    chooseTarget(timestamp);

    if (!finalArmor.size.empty())
    {
        // ����ɹ�ѡ��װ�װ壬���¶�ʧ����������������װ�װ���Ϣ
        finalArmor.center.x += detectRoi.x;
        finalArmor.center.y += detectRoi.y;
        lostCnt = 0;
        lastArmor = finalArmor;
    }
    else
    {
        // ���δ�ɹ�ѡ��װ�װ�

        cout << "lostCnt: " << lostCnt << endl;

        // ���ݶ�ʧ������������һ�ε�װ�װ��С��λ����Ϣ
        if (lostCnt < 8)
            lastArmor.size = Size(lastArmor.size.width * 1.1, lastArmor.size.height * 1.4);
        else if (lostCnt == 9)
            lastArmor.size = Size(lastArmor.size.width * 1.5, lastArmor.size.height * 1.5);
        else if (lostCnt == 12)
            lastArmor.size = Size(lastArmor.size.width * 1.2, lastArmor.size.height * 1.2);
        else if (lostCnt == 15)
            lastArmor.size = Size(lastArmor.size.width * 1.2, lastArmor.size.height * 1.2);
        else if (lostCnt == 18)
            lastArmor.size = Size(lastArmor.size.width * 1.2, lastArmor.size.height * 1.2);
        else if (lostCnt > 33)lastArmor.center = Point2f(0, 0);
    }

    // ��������˺� DRAW_FINAL_ARMOR_MAIN������ԭͼ�ϻ�������װ�װ���ĸ����㣬����ɫ��ʾ
#ifdef DRAW_FINAL_ARMOR_MAIN
    Mat target = src.clone();
    Point2f vertice_armor[4];
    finalArmor.points(vertice_armor);
    for (int i = 0; i < 4; i++) {
        line(target, vertice_armor[i], vertice_armor[(i + 1) % 4], CV_RGB(0, 255, 0));
    }
    imshow("target-show", target);
#endif //DRAW_FINAL_ARMOR_MAIN

    // ��������װ�װ�
    return finalArmor;
}



void ArmorDetector::detectNum(Armor& armor)
{
    Point2f pp[4];
    armor.points(pp);

    Mat numSrc;
    Mat dst;
    _src.copyTo(numSrc);

    //�ҵ��ܿ�ס�������ֵ��ĸ���
    Point2f src_p[4];
    src_p[0].x = pp[1].x + (pp[0].x - pp[1].x) * 1.6;
    src_p[0].y = pp[1].y + (pp[0].y - pp[1].y) * 1.6;

    src_p[1].x = pp[0].x + (pp[1].x - pp[0].x) * 1.6;
    src_p[1].y = pp[0].y + (pp[1].y - pp[0].y) * 1.6;

    src_p[2].x = pp[3].x + (pp[2].x - pp[3].x) * 1.6;
    src_p[2].y = pp[3].y + (pp[2].y - pp[3].y) * 1.6;

    src_p[3].x = pp[2].x + (pp[3].x - pp[2].x) * 1.6;
    src_p[3].y = pp[2].y + (pp[3].y - pp[2].y) * 1.6;


    // ����任
    Mat matrix_per = getPerspectiveTransform(src_p, dst_p);
    warpPerspective(numSrc, dst, matrix_per, Size(30, 60));
    imshow("dst_num", dst);
    dnn_detect(dst, armor);//����������������������������������������������������������������������
}


bool ArmorDetector::conTain(RotatedRect& match_rect, vector<Light>& Lights, size_t& i, size_t& j)
{
    Rect matchRoi = match_rect.boundingRect();
    for (size_t k = i + 1; k < j; k++)
    {
        Point2f lightPs[4];
        // ������ȷ�λ�õĵ�
        if (matchRoi.contains(Lights[k].top) ||
            matchRoi.contains(Lights[k].bottom) ||
            matchRoi.contains(Point2f(Lights[k].top.x + Lights[k].height * 0.25, Lights[k].top.y + Lights[k].height * 0.25)) ||
            matchRoi.contains(Point2f(Lights[k].bottom.x - Lights[k].height * 0.25, Lights[k].bottom.y - Lights[k].height * 0.25)) ||
            matchRoi.contains(Lights[k].center))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

int ArmorDetector::armorGrade(const Armor& checkArmor)
{
    // ��������ϵͳ��ͨ�Ż��ƣ��״�������淶��

    // ѡ����int�ض�double

    /////////id���ȼ������Ŀ////////////////////////
    int id_grade;
    int check_id = checkArmor.id;
    if (check_id == lastArmor.id)
    {
        id_grade = check_id == 1 ? 100 : 80;
    }
    else
    {
        id_grade = check_id == 1 ? 80 : 60;
    }
    id_grade = 100;
    ////////end///////////////////////////////////

    /////////�����ȴ����Ŀ/////////////////////////
    int wh_grade;
    double wh_ratio = checkArmor.size.width / checkArmor.size.height;
    if (checkArmor.type == BIG)
    {
        wh_grade = wh_ratio / big_wh_standard > 1 ? 100 : (wh_ratio / big_wh_standard) * 100;
    }
    else
    {
        wh_grade = wh_ratio / small_wh_standard > 1 ? 100 : (wh_ratio / small_wh_standard) * 100;
    }
    /////////end///////////////////////////////////

    /////////���װ�װ������Ŀ/////////////////////
    // ���װ�װ壬���������һ����׼ֵ���̶����루����3/4�ף���װ�װ��С��Armor.area����Լ�Ƕ��٣��ִ�Сװ�װ壩
    // �ȱ�׼�����100��С����������������������С�ĵó�����ֵ���С
    int height_grade;
    double armor_height = checkArmor.size.height;
    double end_height = (candidateArmors.end() - 1)->size.height;
    double begin_height = candidateArmors.begin()->size.height;
    double hRotation = (armor_height - end_height) / (begin_height - end_height);
    if (candidateArmors.size() == 1)  hRotation = 1;
    height_grade = hRotation * 100;
    //////////end/////////////////////////////////

    ////////����ͼ�����Ĵ����Ŀ//////////////////////
    // �������ģ������������룬�趨��׼ֵ����ͼ��������ͷ�����Ļ���Ĳ���
    int near_grade;
    double pts_distance = POINT_DIST(checkArmor.center, Point2f(_src.cols * 0.5, _src.rows * 0.5));
    near_grade = pts_distance / near_standard < 1 ? 100 : (near_standard / pts_distance) * 100;
    ////////end//////////////////////////////////

    /////////�Ƕȴ����Ŀ//////////////////////////
    // �ǶȲ���
    int angle_grade;
    angle_grade = (90.0 - fabs(checkArmor.angle)) / 90.0 * 100;
    //cout<<fabs(checkArmor.angle)<<"    "<<angle_grade<<endl;  //55~100
    /////////end///////////////////////////////

    // �����ϵ������ϸ���ڣ�
    int final_grade = id_grade * id_grade_ratio +
        wh_grade * wh_grade_ratio +
        height_grade * height_grade_ratio +
        near_grade * near_grade_ratio +
        angle_grade * angle_grade_ratio;


    //cout<<pts_distance<<endl;
//    cout<<wh_grade<<"   "<<height_grade<<"   "<<near_grade<<"   "<<angle_grade<<"    "<<final_grade<<endl;

    return final_grade;

}
bool ArmorDetector::updateSpinScore()
{
    for (auto score = spin_score_map.begin(); score != spin_score_map.end();)
    {
        SpinHeading spin_status;

        //��Status_Map�����ڸ�Ԫ��
        if (spin_status_map.count((*score).first) == 0)
            spin_status = UNKNOWN;
        else
            spin_status = spin_status_map[(*score).first];
        // �����������Ƴ���Ԫ��
        if (abs((*score).second) <= anti_spin_judge_low_thres && spin_status != UNKNOWN)
        {
            spin_status_map.erase((*score).first);
            score = spin_score_map.erase(score);
            continue;
        }

        if (spin_status != UNKNOWN)
            (*score).second = 0.838 * (*score).second - 1 * abs((*score).second) / (*score).second;
        else
            (*score).second = 0.997 * (*score).second - 1 * abs((*score).second) / (*score).second;
        // ��С�ڸ�ֵʱ�Ƴ���Ԫ��
//        cout << "score: " << (*score).second << endl;
        if (abs((*score).second) < 20 || isnan((*score).second))
        {
            spin_status_map.erase((*score).first);
            score = spin_score_map.erase(score);
            continue;
        }
        else if (abs((*score).second) >= anti_spin_judge_high_thres)
        {
            (*score).second = anti_spin_judge_high_thres * abs((*score).second) / (*score).second;
            if ((*score).second > 0)
                spin_status_map[(*score).first] = CLOCKWISE;
            else if ((*score).second < 0)
                spin_status_map[(*score).first] = COUNTER_CLOCKWISE;
        }
        ++score;
    }

    // cout<<"++++++++++++++++++++++++++"<<endl;
    // for (auto status : spin_status_map)
    // {
    //     cout<<status.first<<" : "<<status.second<<endl;
    // }
    return true;
}
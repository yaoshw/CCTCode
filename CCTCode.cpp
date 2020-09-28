//
// Created by yaoshw on 20-9-26.
//

#include <iostream>
#include <vector>
#include <math.h>
#include "CCTCode.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

/*!
 * 默认构造函数，与默认的参数RegularConcentricPatternParam匹配
 */
CCTCode::CCTCode() {
    bit_n_ = 10;
    circle_center_pix_ = 255;
    assist_mat_ = GenerateAssistMatrix(bit_n_);
}

/*!
 *
 * @param bit_n 圆环编码位数
 * @param circle_center_pix 圆环中心的像素值，应为0或255
 */
CCTCode::CCTCode(int bit_n, int circle_center_pix) {
    bit_n_ = bit_n;
    circle_center_pix_ = circle_center_pix;
    assist_mat_ = GenerateAssistMatrix(bit_n_);
}

/*!
 * 构造辅助矩阵，用于后续计算
 * @param bit_n
 * @return
 */
Eigen::MatrixXi CCTCode::GenerateAssistMatrix(int bit_n){
    Eigen::MatrixXi mat(bit_n,bit_n);
    for(int i=0;i<bit_n;i++){
        for(int j=0;j<bit_n;j++){
            int n = (j+(bit_n-1)*i)%bit_n+1;
            mat(i,j) = std::pow(2,n-1);
        }
    }
    return mat;
}

/*!
 * 给定圆环编码位数，得到可用的编码数值集合
 * @param bit_n
 * @return
 */
std::vector<int> CCTCode::GetValidCode(int bit_n){
    std::vector<int> valid_list;
    Eigen::VectorXi binary(bit_n);
    int loc;
    for(int i=0;i<std::pow(2,bit_n);i++){
        if(i>0 && !(i&0b1)){ //如果是偶数直接跳过
            continue;
        }
        //循环移位，获取i的二进制码
        for(int j=0;j<bit_n;j++){
            binary(j) = i & (0b1<<j) ? 1 : 0;//低位在前，高位在后
        }
        (assist_mat_*binary).minCoeff(&loc);
        //只有当loc==0时，其二进制循环移位的最小值才是它本身
        if(loc==0){
            valid_list.emplace_back(i);
        }
    }
    return valid_list;
}

/*!
 * 生成同心圆环标定图案
 * @param markers 图案中圆心坐标
 * @param show_code 是否在图案上显示编码
 * @return 生成的图案
 */
cv::Mat CCTCode::GenRegularConcentricPattern(std::vector<cv::Point2f>& markers, bool show_code){
    int color_bg = circle_center_pix_, color_circle = 255 - circle_center_pix_;
    cv::Mat img = cv::Mat(param_.img_rows, param_.img_cols, CV_8UC3);
    img.setTo(cv::Vec3b(color_bg,color_bg,color_bg));

    int hori_cell_size = param_.img_cols / (param_.pattern_cols + 1);
    int vert_cell_size = param_.img_rows / (param_.pattern_rows + 1);

    std::vector<int> code_list = GetValidCode(bit_n_);
    if(code_list.size()<param_.pattern_cols*param_.pattern_rows){
        std::cout<<"code bit not enough!"<<std::endl;
        return img;
    }

    int cell_size = std::min(hori_cell_size, vert_cell_size);

    int start_x = hori_cell_size;
    int start_y = vert_cell_size;

    int end_x = hori_cell_size*(param_.pattern_cols-1)+start_x;
    int end_y = vert_cell_size*(param_.pattern_rows-1)+start_y;

    int ring_gap = cell_size / (2*param_.concentric_circles+2+1);
    int cirlce_thickness = param_.circle_thickness == 0 ? (ring_gap/2) : param_.circle_thickness;

    if(cirlce_thickness==0){
        std::cout<<"CIRCLES HAD NO THICKNESS!"<<std::endl;
        return img;
    }

    std::vector<int> circle_radius;
    for(int i = 1; i <= param_.concentric_circles; ++i){
        circle_radius.emplace_back(ring_gap*i);
    }
    int anchor_radius = ring_gap*(param_.concentric_circles+1);
    int count = 0;
    double uint_deg = 360/bit_n_;
    markers.clear();
    for(int y = start_y; y <= end_y; y += vert_cell_size){
        for(int x = start_x; x <= end_x; x += hori_cell_size){
            // 按行存储
            markers.emplace_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            for(int j=0;j<bit_n_;j++){
                bool draw = (code_list[count] & (0b1<<j)) !=0;//低位在前，高位在后，顺时针编码画图
                if(draw){
                    cv::ellipse(img,cv::Point(x,y),cv::Size(anchor_radius+cirlce_thickness/2,anchor_radius+cirlce_thickness/2),
                                0,j*uint_deg,(j+1)*uint_deg,cv::Scalar(color_circle,color_circle,color_circle),-1);
                }
            }
            cv::circle(img, cv::Point(x,y), anchor_radius-cirlce_thickness/2, cv::Scalar(color_bg,color_bg,color_bg), -1);
            for(auto& radius:circle_radius){
                cv::circle(img, cv::Point(x,y), radius, cv::Scalar(color_circle,color_circle,color_circle), cirlce_thickness);
            }
            if(show_code) {
                cv::putText(img, std::to_string(code_list[count]), cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX, 2,
                            cv::Scalar(0, 0, 255), 2, 8, 0);
            }
            count ++;
        }
    }

    cv::namedWindow("", cv::WINDOW_NORMAL);
    cv::imshow("",img);
    cv::waitKey();
    cv::destroyAllWindows();
    return img;
}

/*!
 * 对转正拉平的图片解码
 * @param img 输入的已经转正拉平的图像，可以是roi
 * @param center 需要解码的圆心坐标
 * @param radius 内圆内侧的半径
 * @return 解出的编码
 */
int CCTCode::GetRoiDecodeNum(cv::Mat img, cv::Point center, double radius){
    if(img.channels()>1){
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
    int deg_gap = 360/bit_n_;
    Eigen::VectorXi ring_bit(360);
    int center_pix = img.at<uchar>(center);
    //顺时针取出360个采样点
    for(int deg=0;deg<360;deg++){
        int x = center.x+radius*std::cos(deg*M_PI/180);
        int y = center.y+radius*std::sin(deg*M_PI/180);
        ring_bit(deg) = std::abs(img.at<uchar>(y,x)-center_pix)>128 ? 1:0;
    }
    Eigen::Map<Eigen::MatrixXi> binary_set(ring_bit.data(), deg_gap,bit_n_); //解码的二进制数集合
    Eigen::MatrixXi min_binary_set(deg_gap,bit_n_); //循环移位后最小的数的二进制集合
    for(int i=0;i<binary_set.rows();i++){
        int min_code = (assist_mat_*binary_set.row(i).transpose()).minCoeff();
        for(int j=0;j<bit_n_;j++){
            min_binary_set(i,j) = min_code & (0b1<<j) ? 1 : 0;//低位在前，高位在后
        }
    }
    Eigen::VectorXi binary = min_binary_set.colwise().mean(); //对每一位求和然后取平均
//    Eigen::VectorXi binary = min_binary_set.colwise().sum();
//    binary = binary.unaryExpr([&](int d){return d>deg_gap/2? 1:0;});
    return (assist_mat_*binary).minCoeff();
}

/*!
 * 对原始图片解码
 * @param img 原始图片，尚未转正拉平
 * @param point 需要解码的圆心坐标
 * @return 解码结果，如果解码失败，则返回-1
 */
int CCTCode::OneCircleCenterDecode(cv::Mat img, cv::Point point){
    if(img.channels()>1){
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
    cv::threshold(img,img,128,255,cv::THRESH_BINARY);
    int center_pix = img.at<uchar >(point);
    //粗略判断，如果中心点像素不匹配，则返回-1
    if(center_pix!=circle_center_pix_){
        return -1;
    }
    //找到中心椭圆长短轴粗略长度,根据相邻两个点的像素与圆心点不同进行判断
    int hori_r = 0;
    int ver_r = 0;
    for(int i=point.x;i<img.cols-1;i++){
        int d1 = std::abs(img.at<uchar >(point.y,i)-center_pix);
        if(d1==255){
            int d2 = std::abs(img.at<uchar >(point.y,i+1)-center_pix);
            if(d2==255){
                hori_r = i-point.x;
                break;
            }
        }
    }
    for(int i=point.y;i<img.rows-1;i++){
        int d1 = std::abs(img.at<uchar >(i,point.x)-center_pix);
        if(d1==255){
            int d2 = std::abs(img.at<uchar >(i+1,point.x)-center_pix);
            if(d2==255){
                ver_r = i-point.y;
                break;
            }
        }
    }
    //取出roi,3为经验值，与画图时参数有关,实质为根据内圆内侧半径取出整个编码图案
    int len = (hori_r+ver_r)*3;
    if(point.y-len<0 || point.y+len>img.rows-1 || point.x-len<0 || point.x+len>img.cols-1){
        return -1;
    }
    cv::Mat roi = img(cv::Range(point.y-len,point.y+len),cv::Range(point.x-len,point.x+len));

    //获取内圆的椭圆拟合信息
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(roi, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::RotatedRect ellipse;
    bool found_circle = false;
    //找到最内侧的椭圆
    for(auto& contour:contours){
        if(contour.size()<10*M_PI){ //10*M_PI经验值，点数太少，舍弃
            continue;
        }
        double area = cv::contourArea(contour, false);//椭圆面积S=PI*a*b,圆面积S=PI*r*r
        double length = cv::arcLength(contour,true);//椭圆周长L = T(a+b),圆周长L=2*PI*r
        double R0 = 2*std::sqrt(M_PI*area)/(length+1);
        double area_diff_ratio = std::abs(area - M_PI*hori_r*ver_r)/(M_PI*hori_r*ver_r);
        if(R0<0.7){ //0.7和0.2均为经验值，条件可放宽松
            continue;
        }
        if(area_diff_ratio>0.2){
            continue;
        }
        auto ellipse_canditate = cv::fitEllipse(contour);
        //如果满足椭圆中心点限制，可认为找到了内侧圆的椭圆信息
        if(std::abs(ellipse_canditate.center.x-len)<hori_r && std::abs(ellipse_canditate.center.y-len)<ver_r) {
            ellipse = ellipse_canditate;
            found_circle = true;
        }
    }
    if(!found_circle){
        return -1;
    }
    //根据椭圆信息，将roi转正拉平
    cv::Mat rot_mat = cv::getRotationMatrix2D(ellipse.center,ellipse.angle,1.0);
    cv::warpAffine(roi,roi,rot_mat,cv::Size(2*len+1,2*len+1));
    cv::resize(roi,roi,cv::Size((2*len+1)/ellipse.size.width*ellipse.size.height,2*len+1));

    //将roi转正拉平之后，再次寻找内部圆的半径;
    point.x = roi.cols/2;
    point.y = roi.rows/2;
    for(int i=point.x;i<roi.cols-1;i++){
        int d1 = std::abs(roi.at<uchar >(point.y,i)-center_pix);
        if(d1==255){
            int d2 = std::abs(roi.at<uchar >(point.y,i+1)-center_pix);
            if(d2==255){
                hori_r = i-point.x;
                break;
            }
        }
    }
    for(int i=point.y;i<roi.rows-1;i++){
        int d1 = std::abs(roi.at<uchar >(i,point.x)-center_pix);
        if(d1==255){
            int d2 = std::abs(roi.at<uchar >(i+1,point.x)-center_pix);
            if(d2==255){
                ver_r = i-point.y;
                break;
            }
        }
    }

    //利用转正拉平后的图和半径解码
    //TODO：再次判断，找到四个圆环，黑，白，黑，白，遍历圆环像素，可以是10度或5度的间隔，根据平均像素值判断是否存在同心圆，否则返回-1
    //此处根据内圆内侧半径推断编码层圆环半径需由生成规则回去，目前取测试值1.9，或者优化获取内圆内侧半径的方法
    int code_num = GetRoiDecodeNum(roi, point, (ver_r+hori_r)*1.9);
//    cv::cvtColor(roi, roi, cv::COLOR_GRAY2BGR);
//    cv::circle(roi, point, (ver_r + hori_r) * 2, cv::Scalar(0, 0, 255), 1);
//    cv::circle(roi, point, (ver_r + hori_r) * 1.9, cv::Scalar(0, 255, 0), 1);
//    cv::imshow("tmp", roi);
//    cv::waitKey();
    return code_num;
}

/*!
 * 对原始图片解码
 * @param img 原始图片，尚未转正拉平
 * @param markers 需要解码的圆心坐标集合
 * @return 坐标和码值对集合
 */
std::vector<std::pair<cv::Point2f, int>> CCTCode::CircleCentersDecode(cv::Mat img, std::vector<cv::Point2f> markers){
    std::vector<std::pair<cv::Point2f, int>> decoded_markers;
    for(auto& marker:markers){
        int code_num = OneCircleCenterDecode(img,marker);
        decoded_markers.emplace_back(std::make_pair(marker,code_num));
    }
    return decoded_markers;
}

/*!
 * 检测同心圆环的圆心
 * @param img
 * @return
 */
std::vector<cv::Point2f> DetectCircleCenter(cv::Mat img){
    double fitting_rate = 0.9;
    int concentrics=2;
    int bests = 2;
    cv::Mat thresh, blur, edges;
    if(img.channels()>1){
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    }
    cv::threshold(img,thresh,80,255,cv::THRESH_BINARY_INV);
    cv::GaussianBlur(thresh,blur,cv::Size(7,7),0,0);
    cv::GaussianBlur(blur,blur,cv::Size(5,5),0,0);
    cv::GaussianBlur(blur,blur,cv::Size(3,3),0,0);
    cv::Canny(blur,edges,80,120,3,true);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    double limit = M_PI*9;
    cv::Mat canvas;
    std::vector<std::pair<float, cv::Point2f>> marker_candidates;
    std::vector<cv::Point2f> refined_contour;
    std::vector<cv::Point2f> accurate_contour;
    for(int i=0;i<contours.size();i++){
        canvas = cv::Mat::zeros(img.rows,img.cols, CV_8U);
        //TODO: 添加初始过滤条件
        if(contours[i].size()<limit){
            continue;
        }

        auto ellipse = cv::fitEllipse(contours[i]);
        cv::ellipse(canvas,ellipse,cv::Scalar(255));
        refined_contour.clear();
        for(auto& p:contours[i]){
            int px = static_cast<int>(roundf(p.x));
            int py = static_cast<int>(roundf(p.y));
            for(int x = px-1; x <= px+1; ++x){
                for(int y = py-1; y <= py+1; ++y){
                    if(x>=0 && x<canvas.cols && y>=0 && y<canvas.rows){
                        if(canvas.at<uchar>(cv::Point(x,y))==255){
                            refined_contour.emplace_back(cv::Point(x,y));
                            // TODO:加入原始浮点坐标
                        }
                    }
                }
            }
        }
        if(refined_contour.size()<limit){
            continue;
        }

        canvas = cv::Mat::zeros(img.rows,img.cols, CV_8U);
        ellipse = cv::fitEllipse(refined_contour);
        cv::ellipse(canvas, ellipse, cv::Scalar(255));
        accurate_contour.clear();
        for(auto& p:refined_contour){

            int px = static_cast<int>(roundf(p.x));
            int py = static_cast<int>(roundf(p.y));
            for(int x = px-1; x <= px+1; ++x){
                for(int y = py-1; y <= py+1; ++y){
                    if(x>=0 && x<img.cols && y>=0 && y<img.rows){
                        if(canvas.at<uchar>(cv::Point(x,y))==255){
                            accurate_contour.emplace_back(cv::Point(x,y));
                            // TODO:加入原始浮点坐标
                        }
                    }
                }
            }
        }

        if(accurate_contour.size()<limit){
            continue;
        }

        canvas = cv::Mat::zeros(img.rows,img.cols, CV_8U);
        ellipse = cv::fitEllipse(accurate_contour);
        cv::ellipse(canvas, ellipse, cv::Scalar(255));

        int count = 0;
        for(auto& p:accurate_contour){
            if(canvas.at<uchar >(p)==255){
                count++;
            }
        }
        float fit_rate = float(count)/float(accurate_contour.size());
        if(fit_rate >= fitting_rate){
            marker_candidates.emplace_back(std::make_pair(fit_rate, ellipse.center));
        }
    }


    std::vector<cv::Point> cluster_centers;
    std::vector<std::vector<std::pair<float, cv::Point2f>>> clusters;

    cv::Point cluster_center;
    std::vector<std::pair<float, cv::Point2f>> cluster;
    for(auto& marker:marker_candidates){
        if(cluster_centers.empty()){
            cluster_center = cv::Point(int(roundf(marker.second.x)),int(roundf(marker.second.y)));
            cluster_centers.emplace_back(cluster_center);
            cluster.clear();
            cluster.emplace_back(marker);
            clusters.emplace_back(cluster);
        }else{
            auto result = std::find_if(cluster_centers.begin(),cluster_centers.end(), [&marker](const cv::Point& elem){
                return ((std::abs(int(roundf(marker.second.x))-elem.x)<=1)&&(std::abs(int(roundf(marker.second.y))-elem.y)<=1));
            });
            if(result==cluster_centers.end()){
                cluster_center = cv::Point(int(roundf(marker.second.x)),int(roundf(marker.second.y)));
                cluster_centers.emplace_back(cluster_center);
                cluster.clear();
                cluster.emplace_back(marker);
                clusters.emplace_back(cluster);
            }else{
                clusters[static_cast<int>(result-cluster_centers.begin())].emplace_back(marker);
            }
        }
    }

    std::vector<std::vector<std::pair<float, cv::Point2f>>> filtered_clusters;
    for(int i = 0; i < clusters.size(); ++i){
        if(clusters[i].size() >= concentrics){
            filtered_clusters.emplace_back(clusters[i]);
        }
    }
    clusters = filtered_clusters;

    std::vector<cv::Point2f> markers;
    std::map<float, cv::Point2f, std::greater<>> sorted_cluster;
    float marker_x, marker_y;
    for(auto& marker_cluster : clusters){
        marker_x = 0.0;
        marker_y = 0.0;
        sorted_cluster.clear();

        for(auto& point:marker_cluster){
            sorted_cluster.insert(point);
        }

        // 两个同心圆, 取得分最高的两个轮廓
        int select_num = 0;
        for(auto& point:sorted_cluster){
            if(select_num>=bests){
                break;
            }
            marker_x += point.second.x;
            marker_y += point.second.y;
            select_num++;
        }
        marker_x = marker_x/static_cast<float>(select_num);
        marker_y = marker_y/static_cast<float>(select_num);

        cv::Point2f marker = cv::Point2f(marker_x, marker_y);
        markers.emplace_back(marker);
    }
    cv::cvtColor(img,img,cv::COLOR_GRAY2BGR);
    for(auto& marker:markers){
        cv::drawMarker(img,marker,cv::Scalar(0,0,255),cv::MARKER_CROSS,20,3);
    }
    cv::namedWindow("tmp",cv::WINDOW_NORMAL);
    cv::imshow("tmp",img);
    cv::waitKey();
    return markers;
}

/*!
 * 输入图片，检测并解码
 * @param img
 * @return
 */
std::vector<std::pair<cv::Point2f, int>> CCTCode::Decode(cv::Mat img) {
    std::vector<cv::Point2f> centers = DetectCircleCenter(img);
    std::vector<std::pair<cv::Point2f, int>> decode_res = CircleCentersDecode(img,centers);
    if(img.channels()==1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    for(auto& marker:decode_res){
        if(marker.second==-1){
            cv::drawMarker(img,marker.first,cv::Scalar(0,0,255),cv::MARKER_CROSS,20,3);
        }else {
            cv::drawMarker(img, marker.first, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 3);
            cv::putText(img, std::to_string(marker.second), marker.first, cv::FONT_HERSHEY_COMPLEX, 2,
                        cv::Scalar(0, 255, 0), 2, 8, 0);
        }
    }
    cv::namedWindow("tmp",cv::WINDOW_NORMAL);
    cv::imshow("tmp",img);
    cv::waitKey();
    return decode_res;
}
//
// Created by yaoshw on 20-9-26.
//

#ifndef CCTDECODE_CCTCODE_H
#define CCTDECODE_CCTCODE_H

#include "opencv2/opencv.hpp"
#include <Eigen/Core>

struct RegularConcentricPatternParam {
    // 默认是按照A4纸的尺寸生成pattern 打印时只要取消勾选“适应边框打印” 打印出来的同心圆之间的圆心距离就刚好是25mm
    int img_rows=2100;
    int img_cols=2970;
    int pattern_rows=7;
    int pattern_cols=10;
    int circle_thickness=0; // 等于0则自动求算
    int concentric_circles=2;
};
class CCTCode{
public:
    CCTCode();
    CCTCode(int bit_n, int circle_center_pix);
    std::vector<std::pair<cv::Point2f, int>> Decode(cv::Mat img);

    std::vector<int> GetValidCode(int bit_n);
    cv::Mat GenRegularConcentricPattern(std::vector<cv::Point2f>& markers, bool show_code = false);

    std::vector<std::pair<cv::Point2f, int>> CircleCentersDecode(cv::Mat img, std::vector<cv::Point2f> markers);
    int OneCircleCenterDecode(cv::Mat img, cv::Point point);

    void SetBitNum(int bit_n){bit_n_ = bit_n;};
    void SetCenPixValue(int circle_center_pix){circle_center_pix_ = circle_center_pix;};
    int GetBitNum(){return bit_n_;};
    int GetCenPixValue(){return circle_center_pix_;};

    RegularConcentricPatternParam param_;

private:
    Eigen::MatrixXi GenerateAssistMatrix(int bit_n);
    int GetRoiDecodeNum(cv::Mat img, cv::Point center, double radius);

    int bit_n_=10, circle_center_pix_=255;
    Eigen::MatrixXi assist_mat_;

};

#endif //CCTDECODE_CCTCODE_H

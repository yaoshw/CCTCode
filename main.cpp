#include <iostream>
#include "CCTCode.h"

int main() {
    //全部使用默认值
    CCTCode cct;
    std::vector<cv::Point2f> markers;
    cv::Mat img = cct.GenRegularConcentricPattern(markers);
    cct.Decode(img);
    //解码图案
    cv::Mat test_img = cv::imread("../circle.jpg");
    cct.Decode(test_img);

    //自定义
//    cct.SetBitNum(9);
//    cct.SetCenPixValue(0);
//    cct.param_.pattern_rows = 6;
    return 0;
}
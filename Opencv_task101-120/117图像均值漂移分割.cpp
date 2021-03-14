#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/yuan_test.png");
	imshow("input", src);
	Mat dst;
	TermCriteria tc = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.1);
	//彩色图像分割----这个函数严格来说并不是图像的分割，而是图像在色彩层面的平滑滤波，
	//	它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
	pyrMeanShiftFiltering(src, dst, 20, 40, 2, tc);
	imshow("mean shift segementation demo", dst);
	waitKey(0);
	return 0;
}
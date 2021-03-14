#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat dresult, eresult;
	Mat src = imread("photo/lena.jpg");

	// 二值图像
	Mat gray, binary, result;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(9, 9), 0, 0);
	//自适应阈值(其实就是局部阈值法)
	adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 15);
	imshow("binary", binary);

	// 定义结构元素 5x5大小矩形
	Mat se = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	//开运算---先腐蚀后膨胀
	morphologyEx(binary, result, MORPH_OPEN, se);

	// 显示
	imshow("open demo", result);
	waitKey(0);
	return 0;
}
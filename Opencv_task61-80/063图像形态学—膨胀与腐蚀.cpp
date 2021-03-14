#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat dresult, eresult;
	Mat src = imread("photo/coins.jpg");

	// 二值图像
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", binary);

	// 定义结构元素 3x3大小矩形
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	// 膨胀
	dilate(binary, dresult, se, Point(-1, -1), 1, 0);
	// 腐蚀
	erode(binary, eresult, se, Point(-1, -1), 1, 0);

	// 显示
	imshow("dilate", dresult);
	imshow("erode", eresult);
	waitKey(0);
	return 0;
}
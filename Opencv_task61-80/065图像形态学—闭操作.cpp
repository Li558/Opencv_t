#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat dresult, eresult;
	Mat src = imread("photo/cells.png");

	// 二值图像
	Mat gray, binary, result;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", binary);

	// 定义结构元素 5x5大小矩形
	Mat se1 = getStructuringElement(MORPH_RECT, Size(25, 5), Point(-1, -1));
	Mat se2 = getStructuringElement(MORPH_RECT, Size(5, 25), Point(-1, -1));
	
	//闭运算---先膨胀后腐蚀
	morphologyEx(binary, result, MORPH_CLOSE, se1);
	imshow("open demo1", result);
	morphologyEx(result, result, MORPH_CLOSE, se2);

	// 显示
	imshow("open demo2", result);
	waitKey(0);
	return 0;
}
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat basic, exter, inter;
	Mat src = imread("photo/girl.jpg");
	if (src.empty())
	{
		printf("can not find image...\n");
		return -1;
	}
	imshow("input", src);

	// 定义结构元素
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

	// 基本梯度
	morphologyEx(src, basic, MORPH_GRADIENT, se);
	imshow("basic gradient", basic);

	// 外梯度
	morphologyEx(src, exter, MORPH_DILATE, se);
	subtract(exter, src, exter);
	imshow("external gradient", exter);

	// 内梯度
	morphologyEx(src, inter, MORPH_ERODE, se);
	subtract(src, inter, inter);
	imshow("internal gradient", exter);

	waitKey(0);
	return 0;
}
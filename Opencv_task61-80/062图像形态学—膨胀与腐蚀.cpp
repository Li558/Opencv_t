#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat dresult, eresult;
	Mat src = imread("photo/girl.jpg");
	imshow("input", src);
	// 定义结构元素 3x3大小矩形
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	// 膨胀
	dilate(src, dresult, se, Point(-1, -1), 1, 0);
	// 腐蚀
	erode(src, eresult, se, Point(-1, -1), 1, 0);

	// 显示
	imshow("dilate", dresult);
	imshow("erode", eresult);
	waitKey(0);
	return 0;
}
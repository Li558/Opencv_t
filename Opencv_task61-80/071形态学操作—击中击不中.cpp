#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat dresult, eresult;
	Mat src = imread("photo/cross.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	// ��ֵͼ��
	Mat gray, binary, result;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("binary", binary);

	// ����ṹԪ��
	Mat se = getStructuringElement(MORPH_CROSS, Size(11, 11), Point(-1, -1));
	// ���л�����--ǰ��������ʴ����Ľ���
	morphologyEx(binary, result, MORPH_HITMISS, se);

	// ��ʾ
	imshow("hit-and-miss demo", result);
	waitKey(0);
	return 0;
}
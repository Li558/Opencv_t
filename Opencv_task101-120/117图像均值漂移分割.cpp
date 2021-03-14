#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/yuan_test.png");
	imshow("input", src);
	Mat dst;
	TermCriteria tc = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.1);
	//��ɫͼ��ָ�----��������ϸ���˵������ͼ��ķָ����ͼ����ɫ�ʲ����ƽ���˲���
	//	�������к�ɫ�ʷֲ��������ɫ��ƽ��ɫ��ϸ�ڣ���ʴ�������С����ɫ����
	pyrMeanShiftFiltering(src, dst, 20, 40, 2, tc);
	imshow("mean shift segementation demo", dst);
	waitKey(0);
	return 0;
}
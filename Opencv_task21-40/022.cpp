#include <iostream>
#include <opencv2/opencv.hpp>
//void whiteFace(Mat& matSelfPhoto, int alpha, int beta);

using namespace cv;

/*void whiteFace(Mat& matSelfPhoto, int alpha, int beta)
{
	for (int y = 0; y < matSelfPhoto.rows; y++)
	{
		for (int x = 0; x < matSelfPhoto.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				matSelfPhoto.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(matSelfPhoto.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
}
*/


int main()
{
	Mat dst1,dst2,cdst;
	Mat image = imread("real.jpg");

	//��һ�ַ���

	////imshow("������ͼƬ", image);
	///*
	//for (int i = 0; i < image.rows; i++)
	//	{
	//		for (int j = 0; j < image.cols; j++)
	//			{
	//				for (int k = 0; k < 3; k++)
	//					{	int tmp = (uchar)image.at<cv::Vec3b>(i, j)[k] * 1.5 + 20;

	//						if (tmp > 255)
	//						{
	//							image.at<cv::Vec3b>(i, j)[k] = 2 * 255 - tmp;
	//						}
	//						else
	//						{
	//							image.at<cv::Vec3b>(i, j)[k] = tmp;
	//						}

	//					}
	//			}
	//	}
	//	*/

	////bilateralFilter(image, dst1, 30, 35, 15);
	////imshow("������ͼƬ", ds1);
	////pyrMeanShiftFiltering(image, dst2, 10, 50);
	////imshow("������ͼƬ", dst2);
	//Mat dst;

	//int value1 = 3, value2 = 1;     //ĥƤ�̶���ϸ�ڳ̶ȵ�ȷ��

	//int dx = value1 * 5;    //˫���˲�����֮һ  
	//double fc = value1 * 12.5; //˫���˲�����֮һ  
	//int p = 50; //͸����  
	//Mat temp1, temp2, temp3, temp4;

	////˫���˲�  
	//bilateralFilter(image, temp1, dx, fc, fc);

	//temp2 = (temp1 - image + 128);

	////��˹ģ��  
	//GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

	//temp4 = image + 2 * temp3 - 255;

	//dst = (image*(100 - p) + temp4 * p) / 100;
	//dst.copyTo(image);
	////��������
	///*
	//for (int i = 0; i < dst.rows; i++)
	//	{
	//		for (int j = 0; j < dst.cols; j++)
	//			{
	//				for (int k = 0; k < 3; k++)
	//					{	int tmp = (uchar)dst.at<cv::Vec3b>(i, j)[k] * 1.5 + 20;

	//						if (tmp > 255)
	//						{
	//							dst.at<cv::Vec3b>(i, j)[k] = 2 * 255 - tmp;
	//						}
	//						else
	//						{
	//							dst.at<cv::Vec3b>(i, j)[k] = tmp;
	//						}

	//					}
	//			}
	//	}
	//	*/




	
	int height = image.rows;//���src1�ĸ�
	int width = image.cols;//���src1�Ŀ�
	dst1 = Mat::zeros(image.size(), image.type());  //������Ҫ������һ����ԭͼһ����С�Ŀհ�ͼƬ              
	float alpha = 1.5;//�����Աȶ�Ϊ1.5
	float beta = 30;//�������ȼ�50
	//ѭ������������ÿһ�У�ÿһ�е�Ԫ��
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (image.channels() == 3)//�ж��Ƿ�Ϊ3ͨ��ͼƬ
			{
				//�������õ���ԭͼ����ֵ�����ظ�����b,g,r
				float b = image.at<Vec3b>(row, col)[0];//nlue
				float g = image.at<Vec3b>(row, col)[1];//green
				float r = image.at<Vec3b>(row, col)[2];//red
				//��ʼ�������أ��Ա���b,g,r���ı���ٷ��ص��µ�ͼƬ��
				dst1.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst1.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst1.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if (image.channels() == 1)//�ж��Ƿ�Ϊ��ͨ����ͼƬ
			{

				float v = image.at<uchar>(row, col);
				dst1.at<uchar>(row, col) = saturate_cast<uchar>(v*alpha + beta);
			}
		}
	}
	//bilateralFilter(image, dst2, 30, 35, 15);
	//imshow("������ͼƬ", dst2);
	//resize(dst1, cdst, cv::Size(), 2.0, 2.0, cv::INTER_AREA);
	cv::imshow("����ͼ", dst1);







	//�ڶ��ַ���


	//Mat matResult;
	//int bilateralFilterVal = 30;  // ˫��ģ��ϵ��
	//imshow("0000", image);
	//whiteFace(image, 1.1, 68);  // �����Աȶ������ȣ�����2Ϊ�Աȶȣ�����3Ϊ����
	//imshow("1111", image);
	//GaussianBlur(image, image, Size(9, 9), 0, 0); // ��˹ģ����������������
	//imshow("2222", image);
	//bilateralFilter(image, matResult, bilateralFilterVal, // ����ĥƤ
	//	bilateralFilterVal * 2, bilateralFilterVal / 2);
	//imshow("3333", matResult);

	//Mat matFinal;

	// //ͼ����ǿ��ʹ�÷����ڱΣ�Unsharpening Mask��������
	//cv::GaussianBlur(matResult, matFinal, cv::Size(0, 0), 9);
	//cv::addWeighted(matResult, 1.5, matFinal, -0.5, 0, matFinal);
	//imshow("4444", matFinal);




	

	//�����ַ���
	

	

















	
	waitKey(0);
	return 0;
}


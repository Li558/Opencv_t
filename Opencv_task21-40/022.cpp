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

	//第一种方法

	////imshow("处理后的图片", image);
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
	////imshow("处理后的图片", ds1);
	////pyrMeanShiftFiltering(image, dst2, 10, 50);
	////imshow("处理后的图片", dst2);
	//Mat dst;

	//int value1 = 3, value2 = 1;     //磨皮程度与细节程度的确定

	//int dx = value1 * 5;    //双边滤波参数之一  
	//double fc = value1 * 12.5; //双边滤波参数之一  
	//int p = 50; //透明度  
	//Mat temp1, temp2, temp3, temp4;

	////双边滤波  
	//bilateralFilter(image, temp1, dx, fc, fc);

	//temp2 = (temp1 - image + 128);

	////高斯模糊  
	//GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

	//temp4 = image + 2 * temp3 - 255;

	//dst = (image*(100 - p) + temp4 * p) / 100;
	//dst.copyTo(image);
	////调节亮度
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




	
	int height = image.rows;//求出src1的高
	int width = image.cols;//求出src1的宽
	dst1 = Mat::zeros(image.size(), image.type());  //这句很重要，创建一个与原图一样大小的空白图片              
	float alpha = 1.5;//调整对比度为1.5
	float beta = 30;//调整亮度加50
	//循环操作，遍历每一列，每一行的元素
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (image.channels() == 3)//判断是否为3通道图片
			{
				//将遍历得到的原图像素值，返回给变量b,g,r
				float b = image.at<Vec3b>(row, col)[0];//nlue
				float g = image.at<Vec3b>(row, col)[1];//green
				float r = image.at<Vec3b>(row, col)[2];//red
				//开始操作像素，对变量b,g,r做改变后再返回到新的图片。
				dst1.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst1.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst1.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if (image.channels() == 1)//判断是否为单通道的图片
			{

				float v = image.at<uchar>(row, col);
				dst1.at<uchar>(row, col) = saturate_cast<uchar>(v*alpha + beta);
			}
		}
	}
	//bilateralFilter(image, dst2, 30, 35, 15);
	//imshow("处理后的图片", dst2);
	//resize(dst1, cdst, cv::Size(), 2.0, 2.0, cv::INTER_AREA);
	cv::imshow("最终图", dst1);







	//第二种方法


	//Mat matResult;
	//int bilateralFilterVal = 30;  // 双边模糊系数
	//imshow("0000", image);
	//whiteFace(image, 1.1, 68);  // 调整对比度与亮度，参数2为对比度，参数3为亮度
	//imshow("1111", image);
	//GaussianBlur(image, image, Size(9, 9), 0, 0); // 高斯模糊，消除椒盐噪声
	//imshow("2222", image);
	//bilateralFilter(image, matResult, bilateralFilterVal, // 整体磨皮
	//	bilateralFilterVal * 2, bilateralFilterVal / 2);
	//imshow("3333", matResult);

	//Mat matFinal;

	// //图像增强，使用非锐化掩蔽（Unsharpening Mask）方案。
	//cv::GaussianBlur(matResult, matFinal, cv::Size(0, 0), 9);
	//cv::addWeighted(matResult, 1.5, matFinal, -0.5, 0, matFinal);
	//imshow("4444", matFinal);




	

	//第三种方法
	

	

















	
	waitKey(0);
	return 0;
}


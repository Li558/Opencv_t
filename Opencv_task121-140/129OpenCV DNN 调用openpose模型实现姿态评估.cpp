
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
using namespace std;

const int POSE_PAIRS[3][20][2] = {
	{   // COCO body
		{ 1,2 },{ 1,5 },{ 2,3 },
		{ 3,4 },{ 5,6 },{ 6,7 },
		{ 1,8 },{ 8,9 },{ 9,10 },
		{ 1,11 },{ 11,12 },{ 12,13 },
		{ 1,0 },{ 0,14 },
		{ 14,16 },{ 0,15 },{ 15,17 }
	},
	{   // MPI body
		{ 0,1 },{ 1,2 },{ 2,3 },
		{ 3,4 },{ 1,5 },{ 5,6 },
		{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
		{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
	},
	{   // hand
		{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
		{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
		{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
		{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
		{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
	} };

int main(int argc, char **argv)
{
	float thresh = 0.1;
	String modelTxt = "D:/projects/pose_body/hand/pose_deploy.prototxt";
	String modelBin = "D:/projects/pose_body/hand/pose_iter_102000.caffemodel";
	String imageFile = "D:/images/hand.jpg";

	// read the network model
	Net net = readNetFromCaffe(modelTxt, modelBin);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);

	//	load image
	Mat img = imread(imageFile);
	imshow("input", img);

	// º”‘ÿÕ¯¬Á
	Mat inputBlob = blobFromImage(img, 1.0 / 255, Size(368, 368), Scalar(0, 0, 0), false, false);
	net.setInput(inputBlob);
	Mat result = net.forward();
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);

	int midx, npairs;
	int nparts = result.size[1];
	int H = result.size[2];
	int W = result.size[3];

	// find out, which model we have
	if (nparts == 19)
	{   // COCO body
		midx = 0;
		npairs = 17;
		nparts = 18; // skip background
	}
	else if (nparts == 16)
	{   // MPI body
		midx = 1;
		npairs = 14;
	}
	else if (nparts == 22)
	{   // hand
		midx = 2;
		npairs = 20;
	}
	else
	{
		cerr << "there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has " << nparts << " parts." << endl;
		return (0);
	}

	// find the position of the body parts
	vector<Point> points(22);
	for (int n = 0; n < nparts; n++)
	{
		// Slice heatmap of corresponding body's part.
		Mat heatMap(H, W, CV_32F, result.ptr(0, n));
		// 1 maximum per heatmap
		Point p(-1, -1), pm;
		double conf;
		minMaxLoc(heatMap, 0, &conf, 0, &pm);
		if (conf > thresh)
			p = pm;
		points[n] = p;
	}

	// connect body parts and draw it !
	float SX = float(img.cols) / W;
	float SY = float(img.rows) / H;
	for (int n = 0; n < npairs; n++)
	{
		// lookup 2 connected body/hand parts
		Point2f a = points[POSE_PAIRS[midx][n][0]];
		Point2f b = points[POSE_PAIRS[midx][n][1]];

		// we did not find enough confidence before
		if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
			continue;

		// scale to image size
		a.x *= SX; a.y *= SY;
		b.x *= SX; b.y *= SY;

		line(img, a, b, Scalar(0, 200, 0), 2);
		circle(img, a, 3, Scalar(0, 0, 200), -1);
		circle(img, b, 3, Scalar(0, 0, 200), -1);
	}

	imshow("OpenPose-Hand", img);
	waitKey();

	return 0;
}

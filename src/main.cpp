#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

#define method modelPCA
/* run this program using the console pauser or add your own getch, system("pause") or input loop */

Ptr<FaceRecognizer> modelLBP = createLBPHFaceRecognizer();
Ptr<FaceRecognizer> modelPCA = createEigenFaceRecognizer();
Ptr<FaceRecognizer> modelFisher = createFisherFaceRecognizer();

void read_csv(string& fileName, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
	ifstream file(fileName.c_str(),ifstream::in);
	String line, path, label;
	while (getline(file, line)) {
		stringstream lines(line);
		getline(lines, path, separator);
		getline(lines, label);
		if ( !path.empty() && !label.empty()) {
			Mat img = imread(path, 0);
			resize(img, img, Size(100, 100), 0, 0, INTER_LINEAR);
			images.push_back(img);
			labels.push_back(atoi(label.c_str()));
		}
	}
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale)
{
	int i = 0;
	double t = 0;
	
	//建立用于存放人脸的向量容器
	vector<Rect> faces;
	
	//定义一些颜色，用来标示不同的人脸
	const static Scalar colors[] =  { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255)};
		
	//建立缩小的图片，加快检测速度
	//nt cvRound (double value)	//对一个double型的数进行四舍五入，并返回一个整型数！
	Mat gray, smallImg(cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1);
	//转成灰度图像，Harr特征基于灰度图
	cvtColor(img, gray, CV_BGR2GRAY);
	//改变图像大小，使用双线性差值
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	//变换后的图像进行直方图均值化处理
	equalizeHist(smallImg, smallImg);
	
	//程序开始和结束插入此函数获取时间，经过计算求得算法执行时间
	t = (double)cvGetTickCount();
	//检测人脸
	//detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，faces表示检测到的人脸目标序列，1.1表示
	//每次图像尺寸减小的比例为1.1，2表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
	//小都可以检测到人脸),CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的
	//最小最大尺寸
	cascade.detectMultiScale( smallImg, faces, 
		1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		|CV_HAAR_SCALE_IMAGE
		,
		Size(30, 30));
	t = (double)cvGetTickCount() - t;
	
	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
		Point center;
		Scalar color = colors[i % 8];
		int radius;
		Point text_lb = Point(faces[i].x, faces[i].y);
		double aspect_ratio = (double)r->width / r->height;
		if( 0.75 < aspect_ratio && aspect_ratio < 1.3) {
			//标示人脸时在缩小之前的图像上标示，所以这里根据缩放比例换算回去
			center.x = cvRound((r->x + r->width * 0.5) * scale);
			center.y = cvRound((r->y + r->height * 0.5) * scale);
			radius = cvRound((r->width + r->height) * 0.25 * scale);
			circle(img, center, radius, color, 3, 8, 0);
		} else
			rectangle(img, cvPoint(cvRound(r->x * scale), cvRound(r->y * scale)), 
						cvPoint(cvRound((r->x + r->width - 1) * scale), cvRound((r->y + r->height - 1) * scale)), 
						color, 3, 8, 0);
		Mat face_test = gray(faces[i]);
		resize(face_test, face_test, Size(100, 100));
		if (!face_test.empty()) {
			int predict = -1;
			double predicted_confidence = 0.0;
			method->predict(face_test, predict, predicted_confidence);
			string name = "";
			switch (predict){
				case 1: name = "MCY";break;
				case 2: name = "TG";break;
				case 3: name = "SZ";break;
				case 4: name = "GG";break;
				case 5: name = "YRK";break;
				case -1: break;
			}
			putText(img, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
		}
	}
	cv::imshow("result", img);
	waitKey(0);
} 

int main()
{
	String csvPath = "image/data.txt";
	vector<Mat> images;
	vector<int> labels;
	read_csv(csvPath, images, labels);
	
	remove("Model.xml");
	method->train(images, labels);
	method->save("Model.xml");
	method->load("Model.xml");
	
//	VideoCapture cap(0);	//打开默认摄像头
//	if(!cap.isOpened()) {
//		return -1;
//	}
	
	Mat frame;
	CascadeClassifier cascade;
	bool stop = false;
	//训练好的文件名称，放置在可执行文件同目录下
	cascade.load("share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml");
	
	const char *pstrImageName = "image/IMG_3675.JPG";
	IplImage *pSrcImage = cvLoadImage(pstrImageName, CV_LOAD_IMAGE_UNCHANGED);
	frame = Mat(pSrcImage);
	resize(frame, frame, Size(1024, 600), 0, 0, INTER_LINEAR);	
	detectAndDraw(frame, cascade, 1);
	
//	while(!stop) {
//		cap >> frame;
//		detectAndDraw(frame, cascade, 2);
//		if(waitKey(30) >=0)
//			stop = true;
//	}
	return 0;
}

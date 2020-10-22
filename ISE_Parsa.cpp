#include <iostream>
#include "highgui/highgui.hpp"
#include "core/core.hpp"
#include "imgproc/imgproc.hpp"
#include <string>
#include <filesystem>
#include <typeinfo>



#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/opencv.hpp" 



#include<iostream>
#include<sstream>


//#include<opencv2/core/core.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/ml/ml.hpp>





using namespace cv;
using namespace std;
using namespace tesseract;

 namespace fs = std::filesystem;
 using namespace fs;


Mat GreyConvert(Mat RGB) {
	Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);
	for (int i = 0; i < RGB.rows; i++) {
		for (int j = 0; j < RGB.cols * 3; j += 3) {
			Grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
		}
	}
	return Grey;
}


Mat BinaryConvert(Mat Grey,int th) {
	Mat Bin = Mat::zeros(Grey.size() ,CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			if (Grey.at<uchar>(i,j) >= th)
			{
				Bin.at<uchar>(i, j) = 255;
			}
		}
	}

	return Bin;
}


Mat InvertConvert(Mat Grey) {
	Mat Inv = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			Inv.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);

		}
	}
	return Inv;
}




Mat Stepfunction(Mat GG) {
	Mat sf = Mat::zeros(GG.size(), CV_8UC1);
	for (int i = 0; i < GG.rows; i++) {
		for (int j = 0; j < GG.cols; j++) {
			if (GG.at<uchar>(i, j) < 100 || GG.at<uchar>(i, j) > 200) {

				sf.at<uchar>(i, j) = 255;
			}
			
			}
		}
	return sf;
}


Mat Darken(Mat GG) {
	Mat ww = Mat::zeros(GG.size(), CV_8UC1);
	for (int i = 0; i < GG.rows; i++) {
		for (int j = 0; j < GG.cols; j++) {
			if (GG.at<uchar>(i, j) < 100 ) {

				ww.at<uchar>(i, j) = GG.at<uchar>(i,j);
			}
			else
			{
				ww.at<uchar>(i, j) = 100;
			}
		}
	}
	return ww;
}


	
Mat blur(Mat Grey, int neighbour) {
	Mat AvgImage = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbour; i < Grey.rows - neighbour; i++) {
		for (int j = neighbour; j < Grey.cols - neighbour; j++) {
			int sum = 0;
			for (int ii = -neighbour; ii <= neighbour; ii++) {
				for (int jj = -neighbour; jj <= neighbour; jj++) {
					sum += Grey.at<uchar>(i + ii, j + jj);
				}
			}
			
			
			AvgImage.at<uchar>(i, j) = sum/((neighbour * 2 + 1) * (neighbour * 2 + 1));
		}
	}
	return AvgImage;
}



Mat MaxFunc(Mat Grey, int neighbour) {
	Mat MaxImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbour; i < Grey.rows - neighbour; i++) {
		for (int j = neighbour; j < Grey.cols - neighbour; j++) {
			int Maxtemp = 0;
			for (int ii = -neighbour; ii <= neighbour; ii++) {
				for (int jj = -neighbour; jj <= neighbour; jj++) {
					if (Grey.at<uchar>(i + ii, j + jj) > Maxtemp)
					{
						Maxtemp = Grey.at<uchar>(i + ii, j + jj);
					}
					
				}
			}
			MaxImg.at<uchar>(i, j) = Maxtemp;
		}
	}
	
	return MaxImg;
}

Mat MinFunc(Mat Grey, int neighbour) {
	Mat MinImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbour; i < Grey.rows - neighbour; i++) {
		for (int j = neighbour; j < Grey.cols - neighbour; j++) {
			int Mintemp = 256;
			for (int ii = -neighbour; ii <= neighbour; ii++) {
				for (int jj = -neighbour; jj <= neighbour; jj++) {
					if (Grey.at<uchar>(i + ii, j + jj) < Mintemp)
					{
						Mintemp = Grey.at<uchar>(i + ii, j + jj);
					}

				}
			}
			MinImg.at<uchar>(i, j) = Mintemp;
		}
	}

	return MinImg;
}


Mat Edge(Mat Grey, int Th) {
	Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++) {
		for (int j = 1; j < Grey.cols - 1; j++) {
			double AvgL = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
			double AvgR = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
			if (abs(AvgR - AvgL) > Th) {
				EdgeImg.at<uchar>(i, j) = 255;
			}	
		}
	}
	return EdgeImg;
}



Mat Laplacian(Mat Grey) {
	Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++) {
		for (int j = 1; j < Grey.cols - 1; j++) {
			int fvalue = Grey.at<uchar>(i, j) * -4 + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i, j + 1) +
				Grey.at<uchar>(i - 1, j) + Grey.at<uchar>(i + 1, j);
			if (fvalue > 255) {
				fvalue = 255;
			}else if (fvalue<0){
				fvalue = 0;
			}
			EdgeImg.at<uchar>(i, j) = fvalue;
		}
	}
	return EdgeImg;
}

Mat Eqh(Mat Grey) {
	Mat Equlized= Mat::zeros(Grey.size(),CV_8UC1);
	double Pixelvalues[256] = { 0 };
	int ImgSize = (Grey.cols) * (Grey.rows);
	
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			for (int v = 0; v < 256;v++ ) {
				if (Grey.at<uchar>(i,j) == v )  // count the pixels
				{
					Pixelvalues[v]++;
					break;
				}

			}
		
		}
	}    
	for (int v = 0; v < 256; v++) {
		Pixelvalues[v] = Pixelvalues[v] / ImgSize; // real prob
	}	double accp[256] = { 0 };

	for (int v = 0; v < 256; v++) { // acc prob
		double acc = 0;
		for (int q=0;q<=v;q++)
		{
			acc += Pixelvalues[q];
		}
		accp[v] = acc;
	}
	int NewVals[256];
	for (int i = 0; i < 256; i++)
	{
		NewVals[i] = accp[i] * 255;
	}
	for (int i = 0; i < Grey.rows; i++) { // set the values to new image
		for (int j = 0; j < Grey.cols; j++) {
				Equlized.at<uchar>(i, j) = NewVals[Grey.at<uchar>(i, j)];
		}
	}

	return Equlized;
}







Mat EqHist(Mat Grey)
{
	Mat EQImg = Mat::zeros(Grey.size(), CV_8UC1);
	int count[256] = { 0 };
	double prob[256] = { 0.0 };
	double Accprob[256] = { 0.0 };
	int newPixel[256] = { 0 };
	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			count[Grey.at<uchar>(i, j)]++;
		}
	}
	// wrote the code to find prob values
	for (int i = 0; i < 256; i++)
	{
		prob[i] = (double)count[i] / (double)(Grey.rows * Grey.cols);
	}



	// calculate accumulative prob // current prob plus all the previous ones 
	Accprob[0] = prob[0];
	for (int i = 1; i < 256; i++)
		Accprob[i] = Accprob[i - 1] + prob[i];



	// find the new pixel 
	for (int i = 0; i < 256; i++)
	{
		newPixel[i] = 255 * Accprob[i];
	}



	// how to generate the image based on new pixel values 
	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			EQImg.at<uchar>(i, j) = newPixel[Grey.at < uchar>(i, j)];
		}
	}



	return EQImg;


}

Mat Dilation(Mat EdgeImg, int neighbor)
{
	Mat DilatedImg = Mat::zeros(EdgeImg.size(), CV_8UC1);
	for (int i = neighbor; i < EdgeImg.rows - neighbor; i++) // exclude border
	{
		for (int j = neighbor; j < EdgeImg.cols - neighbor; j++)
		{

			for (int ii = -neighbor; ii <= neighbor; ii++)
			{
				for (int jj = -neighbor; jj <= neighbor; jj++)
				{
					// dilation concept 
					if (EdgeImg.at<uchar>(i + ii, j + jj) == 255) {
						DilatedImg.at<uchar>(i, j) = 255;
					}
				}
			}

		}
	}
	return DilatedImg;


}



Mat Erosion(Mat EdgeImg, int neighbor)
{
	Mat ErodedImg = Mat::zeros(EdgeImg.size(), CV_8UC1);
	ErodedImg = EdgeImg.clone();
	for (int i = neighbor; i < EdgeImg.rows - neighbor; i++) // exclude border
	{
		for (int j = neighbor; j < EdgeImg.cols - neighbor; j++)
		{

			for (int ii = -neighbor; ii <= neighbor; ii++)
			{
				for (int jj = -neighbor; jj <= neighbor; jj++)
				{
					
					if (EdgeImg.at<uchar>(i + ii, j + jj) == 0)
						ErodedImg.at<uchar>(i, j) = 0;
				}
			}

		}
	}
	return ErodedImg;

}
Mat FindPlateBackUp(Mat Img, int edge_Th = 50, int dilationVal = 10, int ErosionVal = 1) {
	Mat Plate;

	Mat Imageforsegmentation = Dilation(Erosion(Edge(blur(EqHist(GreyConvert(Img)), 1), edge_Th), ErosionVal), dilationVal).clone();
	vector<vector<Point>> segments;
	vector<Vec4i>hierachy1;
	findContours(Imageforsegmentation, segments, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst = Mat::zeros(GreyConvert(Img).size(), CV_8UC3);
	if (!segments.empty())
	{
		for (int i = 0; i < segments.size(); i++)
		{
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, segments, i, colour, -1, 8, hierachy1);
		}
	}
	imshow("Segmented Image", dst);


	Rect rect_first;
	Scalar blackColor = CV_RGB(0, 0, 0);

	for (int i = 0; i < segments.size(); i++)
	{
		rect_first = boundingRect(segments[i]);
		if (rect_first.y < 0.2 * GreyConvert(Img).cols || rect_first.y > 0.8 * GreyConvert(Img).cols ||
			rect_first.x < 0.3 * GreyConvert(Img).rows || rect_first.x > 0.8 * GreyConvert(Img).rows ||
			rect_first.x + rect_first.height > 0.9 * GreyConvert(Img).rows ||
			rect_first.y + rect_first.width > 0.9 * GreyConvert(Img).cols ||
			rect_first.height / rect_first.width > 0.8 || rect_first.height > 120 || rect_first.width < 80)
		{
			drawContours(Imageforsegmentation, segments, i, blackColor, -1, 8, hierachy1);
		}
		else
			Plate = GreyConvert(Img(rect_first));

	}

	imshow("Filtered Image", Imageforsegmentation);


	if (Plate.cols == 0 || Plate.rows == 0) {
		cout << endl << "something is not right";
	}
	else
	{
		imshow("Final Plate", Plate);
	}
	return Plate;
}

void Test(Mat uu) {
	Mat Imageforsegmentation = uu.clone();
	vector<vector<Point>> segments;
	vector<Vec4i>hierachy1;
	findContours(Imageforsegmentation, segments, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst = Mat::zeros(GreyConvert(uu).size(), CV_8UC3);
	if (!segments.empty())
	{
		for (int i = 0; i < segments.size(); i++)
		{
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, segments, i, colour, -1, 8, hierachy1);
		}
		Rect rr = boundingRect(segments[0]);
		cout <<endl<<" h/w: " << float(rr.height)/float(rr.width)<< "  H : " <<rr.height << " W: " << rr.width;
	}
}

Mat FindPlate(Mat Img, int edge_Th = 50, int dilationVal = 10, int ErosionVal = 1, bool Filterbysize = true) {
	Mat Plate ;

	Mat Imageforsegmentation = Dilation(Erosion(Edge(blur(EqHist(GreyConvert(Img)),1), edge_Th), ErosionVal), dilationVal).clone();
	vector<vector<Point>> segments;
	vector<Vec4i>hierachy1;
	findContours(Imageforsegmentation, segments, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst = Mat::zeros(GreyConvert(Img).size(), CV_8UC3);
	if (!segments.empty())
	{
		for (int i = 0; i < segments.size(); i++)
		{
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, segments, i, colour, -1, 8, hierachy1);
		}
	}
	//imshow("Segmented Image", dst);

	
	Rect rect_first;
	Scalar blackColor = CV_RGB(0, 0, 0);

	for (int i = 0; i < segments.size(); i++)
	{
		rect_first = boundingRect(segments[i]);
		if (rect_first.y < 0.2 * GreyConvert(Img).cols || rect_first.y > 0.8 * GreyConvert(Img).cols ||
			rect_first.x < 0.3 * GreyConvert(Img).rows || rect_first.x > 0.95 * GreyConvert(Img).rows ||
			rect_first.x +rect_first.width > 0.9 * GreyConvert(Img).cols || ( Filterbysize &&
			
			((float(rect_first.height) /float(rect_first.width)) > 0.65 || rect_first.height > 120 || rect_first.width < 80)))
		{
			drawContours(Imageforsegmentation, segments, i, blackColor, -1, 8, hierachy1);
		}
		else
			Plate = GreyConvert(Img(rect_first));
		
	}
	
	//imshow("Filtered Image", Imageforsegmentation);
	//Test(Imageforsegmentation);

	if (Plate.cols == 0 || Plate.rows == 0) {
	//	cout << endl << "something is not right";	
	}
	else
	{
	imshow("Final Plate", Plate);
	}
	return Plate;
}

Mat ReadPlate(Mat Org) {
	Mat Plate = FindPlate(Org, 40, 11, 1);
	//cout << endl << "attempt 1";
	if (Plate.cols == 0 || Plate.rows == 0) {
	//	cout << endl << "attempt 2";
		Plate = FindPlate(Org, 60, 9, 1);

		if (Plate.cols == 0 || Plate.rows == 0) {
		//	cout << endl << "attempt 3";
			Plate = FindPlate(Org, 20, 13, 1);

			if (Plate.cols == 0 || Plate.rows == 0) {
		//		cout << endl << "attempt 4";
				Plate = FindPlate(Org, 40, 9, 1);

				if (Plate.cols == 0 || Plate.rows == 0) {
			//		cout << endl << "attempt 5";
					Plate = FindPlate(Org, 40, 11, 1, false);
				}
			}
		}
	}
	return Plate;
}

int OTSU(Mat Grey)
{
	int count[256] = { 0 };
	double prob[256] = { 0.0 };
	double thetha[256] = { 0.0 };
	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			count[Grey.at<uchar>(i, j)]++;
		}
	}
	
	for (int i = 0; i < 256; i++)
	{
		prob[i] = (double)count[i] / (double)(Grey.rows * Grey.cols);
	}



	
	thetha[0] = prob[0];
	for (int i = 1; i < 256; i++)
		thetha[i] = thetha[i - 1] + prob[i];



	
	double meu[256] = { 0.0 };



	for (int i = 1; i < 256; i++)
		meu[i] = meu[i - 1] + i * prob[i];

	// accumulative i * prob[i]



	double sigma[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		sigma[i] = pow(meu[255] * thetha[i] - meu[i], 2) / (thetha[i] * (1 - thetha[i]));



	// find i which has the maximum sigma 
	double iMax = -1000;
	int OtsuValue = 0;
	for (int i = 0; i < 256; i++)
	{
		if (sigma[i] > iMax)
		{
			iMax = sigma[i];
			OtsuValue = i;
		}
	}




	return OtsuValue + 30;


}






Mat GetLetters(Mat Plate) {

	imshow("OTSO", ( BinaryConvert(Plate, OTSU(Plate))));

	Mat Plateforsegmentation;
	Plateforsegmentation = BinaryConvert(Plate, OTSU(Plate)).clone();
	vector<vector<Point>> segments2;
	vector<Vec4i>hierachy2;
	findContours(Plateforsegmentation, segments2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst2 = Mat::zeros(Plate.size(), CV_8UC3);
	if (!segments2.empty())
	{
		for (int i = 0; i < segments2.size(); i++)
		{
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst2, segments2, i, colour, -1, 8, hierachy2);
		}
	}

	Rect rect_first2;

	for (int i = 0; i < segments2.size(); i++)
	{
		Mat character;
		rect_first2 = boundingRect(segments2[i]);
		Scalar blackColor = CV_RGB(0, 0, 0);
		if (float(rect_first2.height) / float(rect_first2.width)< 0.35 || rect_first2.height<2 || rect_first2.width < 2)
			drawContours(Plateforsegmentation, segments2, i, blackColor, -1, 8, hierachy2);
		else
		{
			character = Plate(rect_first2);
		//	imshow("character" + i  , character);

		}

	}
	imshow("segmented plate", dst2);
	imshow("Filtered Plate", Plateforsegmentation);

	//waitKey();

	return Plateforsegmentation;

}


void TranslatePlate(Mat FP) {

	imshow("sent image", FP);
	TessBaseAPI* ocr = new TessBaseAPI();
	ocr->Init("H:\\2nd user\\OpenCV_Root\\ISE_Parsa\\x64\\Release\\tessdata", "eng", OEM_LSTM_ONLY);
	
	ocr->SetPageSegMode(PSM_AUTO);
	ocr->SetVariable("tessedit_char_whitelist", "QWERTYUIOPASDFGHJKLZXCVBNM0123456789");
	//ocr->SetVariable("tessedit_char_blacklist", "-_![]{}/|\@#$%^&*");
	ocr->SetImage(FP.data, FP.cols, FP.rows,FP.channels(), FP.step);
	string outText = string(ocr->GetUTF8Text());

	cout << "plate is : " << outText << endl;

}


int main()
{
	int C;
	path dirs[19];
	int F_Ind = 0;
	string p = "H:\\2nd user\\Desktop2\\Dataset";
	for (auto dir : directory_iterator(p)) {
		
			dirs[F_Ind] = dir.path();
			F_Ind++;
		
		
	}

	
	int C_index = 0;
	for (int i=C_index;i<=19;i++)
	{
		
		
			Mat Org = imread((dirs[i]).u8string());

			cout << (dirs[i]).u8string() << endl;
			imshow("original", Org);

			Mat tempp = (ReadPlate((Org)));
			Mat tempp2 = BinaryConvert(tempp, OTSU(tempp));
			Mat tempp3 = InvertConvert(tempp2);
			TranslatePlate(tempp3);

			waitKey(15);
			C = 0;
			cout << "Press Any Key";
			cin >> C;
	}
	
	

	


}
	


	

	


	
	/* KNN OCR METHOD

			std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
			std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

																		// read in training classifications ///////////////////////////////////////////////////

			cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

			cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

			if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
				std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
				return(0);                                                                                  // and exit program
			}

			fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
			fsClassifications.release();                                        // close the classifications file

																				// read in training images ////////////////////////////////////////////////////////////

			cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

			cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

			if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
				std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
				return(0);                                                                              // and exit program
			}

			fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
			fsTrainingImages.release();                                                 // close the traning images file

																						// train //////////////////////////////////////////////////////////////////////////////

			cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

																						// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
																						// even though in reality they are multiple images / numbers
			kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

			// test ///////////////////////////////////////////////////////////////////////////////

			cv::Mat matTestingNumbers = ReadPlate(Org);// GetLetters(ReadPlate(Org));            // read in the test numbers image

			if (matTestingNumbers.empty()) {                                // if unable to open image
				std::cout << "error: image not read from file\n\n";         // show error message on command line
				return(0);                                                  // and exit program
			}

			cv::Mat matGrayscale;           //
			cv::Mat matBlurred;             // declare more image variables
			cv::Mat matThresh;              //
			cv::Mat matThreshCopy;          //

			matGrayscale = GreyConvert(matTestingNumbers);        // convert to grayscale

																				// blur
			cv::GaussianBlur(matGrayscale,              // input image
				matBlurred,                // output image
				cv::Size(5, 5),            // smoothing window width and height in pixels
				0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

										   // filter image from grayscale to black and white
			cv::adaptiveThreshold(matBlurred,                           // input image
				matThresh,                            // output image
				255,                                  // make pixels that pass the threshold full white
				cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
				cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
				11,                                   // size of a pixel neighborhood used to calculate threshold value
				2);                                   // constant subtracted from the mean or weighted mean

			matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

			std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
			std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

			cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
				ptContours,                             // output contours
				v4iHierarchy,                           // output hierarchy
				cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
				cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

			for (int i = 0; i < ptContours.size(); i++) {               // for each contour
				ContourWithData contourWithData;                                                    // instantiate a contour with data object
				contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
				contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
				contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
				allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
			}

			for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
				if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
					validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
				}
			}
			// sort contours from left to right
			std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

			std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

			for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour

																				// draw a green rect around the current char
				cv::rectangle(matTestingNumbers,                            // draw rectangle on original image
					validContoursWithData[i].boundingRect,        // rect to draw
					cv::Scalar(0, 255, 0),                        // green
					2);                                           // thickness

				cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

				cv::Mat matROIResized;
				cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

				cv::Mat matROIFloat;
				matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

				cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

				cv::Mat matCurrentChar(0, 0, CV_32F);

				kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

				float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

				strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
			}

			std::cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";       // show the full string

			cv::imshow("matTestingNumbers", matTestingNumbers);     // show input image with green boxes drawn around found digits */

			
	


		//TranslatePlate(FP);

		//	waitKey();

		
	






// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>


// OpenCv headers
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <bitset>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <assert.h>
#include <opencv2/flann/flann.hpp>
#include "ANMS_SSC.hpp"

#include "ldb.h"
//#include "LATCHK.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

#include "baft.h"

#include "utils.h"
#include "bold.h"
#include "helper.h"

#include "locky.h"

#include "gms_matcher.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

static const float nn_ratio_threshold = 0.8f;

double calc_detection(Ptr<FeatureDetector> detector, Mat &img, vector<KeyPoint> &keypoints, bool gettime)
{
	double t1,t2,tdet=0;

	if (gettime)
	{
		detector->detect(img, keypoints);
		for(int i=0; i<30;i++){
			t1 = cv::getTickCount();
			detector->detect(img, keypoints);
			t2 = cv::getTickCount();
			tdet += 1000.0*(t2-t1) / cv::getTickFrequency();
			
		}

		tdet/=30;
		cout <<"Cantidad de Keypoints: "<< keypoints.size() << endl;
		cout << "Tiempo deteccion: " << tdet << " ms" << endl;
	}
	else
		detector->detect(img, keypoints);

	if(gettime) return tdet;
	else return 0;
}

double calc_description(Ptr<Feature2D> extractor, Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors, bool gettime)
{
	double t1,t2,tdesc=0;

	if (gettime) {
		extractor->compute(img, keypoints, descriptors);
		for(int i=0; i<30;i++){
			t1 = cv::getTickCount();
			extractor->compute(img, keypoints, descriptors);
			t2 = cv::getTickCount();
			tdesc += 1000.0*(t2-t1) / cv::getTickFrequency();
		}
		tdesc/=30;
		cout<<"Tiempo de descripcion: "<<tdesc<<" ms"<<endl;
	}
	else
		extractor->compute(img, keypoints, descriptors);

	cout<<"Descriptor size: "<<descriptors.size()<<endl;

	if(gettime) return tdesc;
	else return 0;
}

int main(int argc, char** argv)
{
	constexpr bool multithread = true;
	double t1,t2,tdet,tdesc=0,tmatch=0;
	Mat src_1,src_2, descriptors_1, descriptors_2;
	Helper ImageHelper;
	Mat masks_1, masks_2;
	vector<Mat> patches_1, patches_2;

	vector<KeyPoint> keypoints_1, keypoints_2;
	int kpts;
	//src_1 = imread("images/imagenes_blackbird/left_images/left_1531254018.493000.png", CV_LOAD_IMAGE_GRAYSCALE);
	//src_2 = imread("images/imagenes_blackbird/left_images/left_1531254018.588000.png", CV_LOAD_IMAGE_GRAYSCALE);
	//src_1 = imread("/home/emiliano/DRINK/data/leuven/img1.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	//src_2 = imread("/home/emiliano/DRINK/data/leuven/img3.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	src_1 = imread("images/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
	src_2 = imread("images/000001.png", CV_LOAD_IMAGE_GRAYSCALE);
	// src_1 = imread("images/python1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	// src_2 = imread("images/python2.jpg", CV_LOAD_IMAGE_GRAYSCALE);


	//Ayuda
	if(argc < 4 || argc > 6){
		cout<<"Modo de uso: ./test <detector> <descriptor> <matcher> show anms (show y anms son opcionales)"<<endl;
		cout<<"Detector: FAST, ORB, GFTT, AGAST, BRISK, BAFT, LOCKY, LOCKYS"<<endl;
		cout<<"Descriptor: BRIEF, BRISK, FREAK, ORB, LDB, LATCH, LATCHK, BAFT, BOLD ,(los que probemos)"<<endl;
		cout<<"Matcher: BFM, GMS, FLANN"<<endl;

		return 0;
	}


	if( !strcmp("AKAZE", argv[1] )){
		 /*Parametros SPTAM: descriptor_type: 5 descriptor_size: 0 descriptor_channels: 1 threshold: 0.003 nOctaves: 1 			nOctaveLayers: 1 diffusivity: 1*/
		Ptr<FeatureDetector> detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB,0,1,0.003f,1,1);
		detector->detect(src_1,keypoints_1,Mat());
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		//cout<< tdet<<endl;
		calc_detection(detector, src_2, keypoints_2, false);
	}

	//Si el detector es FAST
	else if( !strcmp("FAST", argv[1] )){
		/* void FAST(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression=true )*/
		/* Parametros SPTAM:   threshold: 60 nonmaxSuppression: true*/

		Ptr<FastFeatureDetector> detector=FastFeatureDetector::create(20, true);
		//Ptr<FastFeatureDetector> detector_2=FastFeatureDetector::create(106);
		detector->detect(src_1,keypoints_1,Mat());
		
		/*vector<cv::KeyPoint> sscKP = */
		tdet=calc_detection(detector, src_1, keypoints_1, true);

		/*ANMS*/
		/*double t1, t2, tanms=0;
		t1 = cv::getTickCount();
		ssc(keypoints_1,500,0.1,src_1.cols,src_1.rows);
		t2 = cv::getTickCount();
		tanms=1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo ANMS: "<<tanms<<endl;
		cout<<"Tiempo det + ANMS: "<<tanms+tdet<<endl;*/

		calc_detection(detector, src_2, keypoints_2, false);

		//cout<< tdet<<endl;
		//ssc(keypoints_1,500,0.1,src_1.cols,src_1.rows);
		//ssc(keypoints_2,500,0.1,src_2.cols,src_2.rows);
	}

	//Si el detector es ORB
	else if( !strcmp("ORB", argv[1] )){
		/*Parametros por defecto:
		ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0,
		int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)*/
		/*Parametros SPTAM:    nFeatures: 2000 scaleFactor: 1.2  nLevels: 1 edgeThreshold: 31*/
		Ptr<FeatureDetector> detector = ORB::create(5000,1.2,1,31);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);

		/*double t1,t2, tanms=0;
		t1 = cv::getTickCount();
		ssc(keypoints_1,500,0.1,src_1.cols,src_1.rows);
		t2 = cv::getTickCount();
		tanms=1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo ANMS: "<<tanms<<endl;
		cout<<"Tiempo det + ANMS: "<<tanms+tdet<<endl;*/
		

		calc_detection(detector, src_2, keypoints_2, false);
	}

	//Si el detector es BRISK
	else if( !strcmp("BRISK", argv[1] )){
		/*Parametros SPTAM   # orientationNormalized: 'true'# scaleNormalized: 'true'# patternScale: '22.0'# OpenCV3
  		thresh: 80 octaves: 1 patternScale: 1.0

		*/

		Ptr<FeatureDetector> detector = BRISK::create(10,1,1.0);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

	//Si el detector es AGAST
	else if( !strcmp("AGAST", argv[1] )){
		/*Parametros por defecto:
		  cv::AgastFeatureDetector::create(int threshold = 10, bool nonmaxSuppression = true,
		  int type = AgastFeatureDetector::OAST_9_16)
		 */

		/* Parametros SPTAM:  threshold: 60 nonmaxSuppression: true  */
		Ptr<FeatureDetector> detector = AgastFeatureDetector::create(30, false);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);

		/*double t1,t2, tanms=0;
		t1 = cv::getTickCount();
		ssc(keypoints_1,500,0.1,src_1.cols,src_1.rows);
		t2 = cv::getTickCount();
		tanms=1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo ANMS: "<<tanms<<endl;
		cout<<"Tiempo det + ANMS: "<<tanms+tdet<<endl;*/
			

		calc_detection(detector, src_2, keypoints_2, false);
		//ssc(keypoints_2,500,0.1,src_2.cols,src_2.rows);
	}

	//Si el detector es GFTT
	else if( !strcmp("GFTT", argv[1] )){
		/*Parametros por defecto: static Ptr< GFTTDetector > 	create (int maxCorners=1000, double qualityLevel=0.01,
		  double minDistance=1, int blockSize=3, bool useHarrisDetector=false, double k=0.04)*/

  		/*Parametros SPTAM : nfeatures: 1000  qualityLevel: 0.01 minDistance: 15.0   useeHarrisDetector: false*/

		//tarda lo mismo en detectar 500 o 1000 puntos
		Ptr<FeatureDetector> detector = GFTTDetector::create(500,0.01,16.1,3,false,0.04);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

		//Si el detector es BAFT
	else if( !strcmp("BAFT", argv[1] )){
		/*Parametros por defecto:
		ORB(int nfeatures=500, float scaleFact or=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0,
		int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)*/

		/* Parametros SPTAM    nfeatures: 500 size: 20 patchSize: 16 #30 gaussianBlurSize: 0 fullRotation: true
  		scaleFactor: 1.2 nlevels: 4 #8 edgeThreshold: 45 fastThreshold: 20 #20*/

		Ptr<Feature2D> detector = BAFT::create(5000,16,20,0,true,1.2,4,45,20);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		
		/*double t1,t2, tanms=0;
		t1 = cv::getTickCount();
		ssc(keypoints_1,500,0.1,src_1.cols,src_1.rows);
		t2 = cv::getTickCount();
		tanms=1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo ANMS: "<<tanms<<endl;
		cout<<"Tiempo det + ANMS: "<<tanms+tdet<<endl;*/

		calc_detection(detector, src_2, keypoints_2, false);
		//ssc(keypoints_2,500,0.1,src_2.cols,src_2.rows);
	}

	else if (!strcmp("LOCKYS", argv[1] )) {
		cv::Ptr<locky::LOCKYFeatureDetector> detector = locky::LOCKYFeatureDetector::create(100000,7,3,20,true);
		detector->detect(src_1, keypoints_1);
		t1 = cv::getTickCount();
		detector->detect(src_1, keypoints_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		detector->detect(src_2, keypoints_2);
		cout <<"Cantidad de Keypoints: "<< keypoints_1.size() << endl;
		cout << "Tiempo deteccion: " << tdet << " ms" << endl;
	}

	else if (!strcmp("LOCKY", argv[1] )) {
		cv::Ptr<locky::LOCKYFeatureDetector> detector = locky::LOCKYFeatureDetector::create(45000,5,3,30,false); //100000,7,3,30
		detector->detect(src_1, keypoints_1);
		t1 = cv::getTickCount();
		detector->detect(src_1, keypoints_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		detector->detect(src_2, keypoints_2);
		cout <<"Cantidad de Keypoints: "<< keypoints_1.size() << endl;
		cout << "Tiempo deteccion: " << tdet << " ms" << endl;
	}

	else{
		cout<<argv[1]<<" no es un nombre de detector valido"<<endl;
		cout<<"Detectores: FAST, ORB, GFTT, AGAST, BRISK, BAFT, LOCKY, LOCKYS"<<endl;
		return 0;
	}

	//kpts=keypoints_1.size();
        //cout<<"argc: "<<argc<<endl;
	//cout<<argv[5]<<endl;
	if((argc == 6 && !strcmp("anms", argv[5] )) || (argc == 5 && !strcmp("anms", argv[4] )) ){
		/*ANMS*/
		double t1,t2, tanms=0;
		t1 = cv::getTickCount();
		ssc(keypoints_1,500,0.1,src_1.cols,src_1.rows);
		t2 = cv::getTickCount();
		tanms=1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo ANMS: "<<tanms<<endl;
		cout<<"Tiempo det + ANMS: "<<tanms+tdet<<endl;
                ssc(keypoints_2,500,0.1,src_2.cols,src_2.rows);

	}

	//Descriptores
	//Si el descriptor es BRIEF

	if( !strcmp("BRIEF", argv[2] )){
		/*Parametros por defecto:
		  static Ptr< BriefDescriptorExtractor > 	create (int bytes=32, bool use_orientation=false)*/
		
		/*parametros SPTAM  bytes=32 */
		Ptr<BriefDescriptorExtractor> featureExtractor = BriefDescriptorExtractor::create(32,false);
		//Ptr<BriefDescriptorExtractor> featureExtractor_2 = BriefDescriptorExtractor::create();

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es BRISK
	else if( !strcmp("BRISK", argv[2] )){
		/*Parametros por defecto:
		  BRISK::BRISK(int thresh=30, int octaves=3, float patternScale=1.0f)*/
		/*Parametros SPTAM   # orientationNormalized: 'true'# scaleNormalized: 'true'# patternScale: '22.0'# OpenCV3
  		thresh: 80 octaves: 1 patternScale: 1.0*/

		Ptr<Feature2D> featureExtractor = BRISK::create(115,1,1.0);

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es FREAK
	else if( !strcmp("FREAK", argv[2] )){

		/*static Ptr< FREAK > 	create (bool orientationNormalized=true, bool scaleNormalized=true,
		  float patternScale=22.0f, int 	nOctaves=4, const std::vector< int > &selectedPairs=std::vector< int >())*/

		/*Parametros SPTAM:   patternScale: 42.0 orientationNormalized: true scaleNormalized: true nOctaves: 8 */
		Ptr<Feature2D> featureExtractor = FREAK::create(42.0,true,8);
		//extractor->compute(src_1, keypoints_1, descriptors_1);
		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es ORB
	else if( !strcmp("ORB", argv[2] )){
		/*static Ptr< ORB >	create (int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int
		  firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int
		  fastThreshold=20)*/

		/*Parametros SPTAM:  Name: 'ORB' nFeatures: 2000 scaleFactor: 1.2 nLevels: 1 edgeThreshold: 31*/
		Ptr<Feature2D> featureExtractor = ORB::create(500,1.2,1,31);

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es AKAZE
	else if( !strcmp("AKAZE", argv[2] )){
		/*static Ptr< ORB >	create (int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int
		  firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int
		  fastThreshold=20)*/

		/*Parametros SPTAM: descriptor_type: 5 descriptor_size: 0 descriptor_channels: 1 threshold: 0.003 nOctaves: 1 			nOctaveLayers: 1 diffusivity: 1*/
		Ptr<Feature2D> featureExtractor = AKAZE::create(AKAZE::DESCRIPTOR_MLDB,0,3,0.003f,1,1);

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es LATCH
	else if( !strcmp("LATCH", argv[2] )){
		/*static Ptr<LATCH> cv::xfeatures2d::LATCH::create(int bytes = 32,
		 *bool rotationInvariance = true, int half_ssd_size = 3, double sigma = 2.0)*/		
		/*Parametros SPTAM   bytes: 32 rotationInvariance: false half_ssd_size: 3*/
		Ptr<Feature2D> featureExtractor = LATCH::create(32,3);

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}
/*
	// ------------- LATCHK ------------
	else if( !strcmp("LATCHK", argv[2] )){
		uint64_t* desc_1 = new uint64_t[8 * keypoints_1.size()];
		std::vector<KeyPointK> kps1;
		for (auto&& kp : keypoints_1) kps1.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);
		LATCHK<multithread>(src_1.data, src_1.cols, src_1.rows, static_cast<int>(src_1.step), kps1, desc_1);
		t1 = cv::getTickCount();
		for (int i=0; i<30; i++)
			LATCHK<multithread>(src_1.data, src_1.cols, src_1.rows, static_cast<int>(src_1.step), kps1, desc_1);
		for (size_t i=0; i < 8 * kps1.size(); ++i)
			desc_1[i] =  __builtin_bswap64 (desc_1[i]);
		descriptors_1 = Mat(keypoints_1.size(), 64, CV_8U, desc_1, 64);
		t2 = cv::getTickCount();
		tdesc = 1000.0*(t2-t1) / cv::getTickFrequency() / 30;
		cout<<"Tiempo de descripcion: "<<tdesc<<" ms"<<endl;
		// --------------------------------

		// ------------- LATCHK ------------
		uint64_t* desc_2 = new uint64_t[8 * keypoints_2.size()];
		std::vector<KeyPointK> kps2;
		for (auto&& kp : keypoints_2) kps2.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);
		LATCHK<multithread>(src_2.data, src_2.cols, src_2.rows, static_cast<int>(src_2.step), kps2, desc_2);
		LATCHK<multithread>(src_2.data, src_2.cols, src_2.rows, static_cast<int>(src_2.step), kps2, desc_2);
		for (size_t i=0; i < 8 * kps2.size(); ++i)
			desc_2[i] =  __builtin_bswap64 (desc_2[i]);
		descriptors_2 = Mat(keypoints_2.size(), 64, CV_8U, desc_2, 64);
		// -------------------------------- 
	}
*/
	//Si el descriptor es LDB
	else if( !strcmp("LDB", argv[2] )){
		//LDB(int _bytes = 32, int _nlevels = 3, int _patchSize = 60);
		//Feature2D featureExtractor = LdbDescriptorExtractor::create();

		//LDB featureExtractor(32); //48
		LDB featureExtractor(32);
		featureExtractor.compute(src_1, keypoints_1, descriptors_1, 0);
		t1 = cv::getTickCount();
		for(int i=0;i<30;i++)
			featureExtractor.compute(src_1, keypoints_1, descriptors_1, 0);
		t2 = cv::getTickCount();
		tdesc = 1000.0*(t2-t1) / cv::getTickFrequency() / 30;
		cout <<"Cantidad de Keypoints: "<< keypoints_1.size() << endl;
		cout << "Tiempo descripcion: " << tdesc << " ms" << endl;
		featureExtractor.compute(src_2, keypoints_2, descriptors_2, 0);
		cout<<"Descriptor size: "<<descriptors_1.size()<<endl;
		//calc_description(&featureExtractor, src_1, keypoints_1, descriptors_1, true);
		//calc_description(&featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//si el descriptor es BAFT
	else if(!strcmp("BAFT", argv[2])){

		/*Parametros por defecto:
    		CV_WRAP static Ptr<BAFT> create(int nfeatures=1000, int size=128,
            	int patchSize=30, int gaussianBlurSize = 0, bool fullRotation=0,
            	float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=45, int fastThreshold=20);*/

		/* Parametros SPTAM    nfeatures: 600 size: 24 patchSize: 8 #30 gaussianBlurSize: 0 fullRotation: false
  		scaleFactor: 1.2 nlevels: 8 #8 edgeThreshold: 45 fastThreshold: 20 #20*/
		Ptr<Feature2D> featureExtractor = BAFT::create(5000,24,30,0,true,1.2,4,45,20);
		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//si el descriptor es BINBOOST
	else if(!strcmp("BINBOOST", argv[2])){
		Ptr<Feature2D> featureExtractor = BoostDesc::create(BoostDesc::BINBOOST_64);
		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//si el descriptor es VGG
	else if(!strcmp("VGG", argv[2])){
		Ptr<Feature2D> featureExtractor = VGG::create(VGG::VGG_120, 1.4f, true, true, 5.00f, false);
		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//si el descriptor es BOLD
	else if( !strcmp("BOLD", argv[2])) {
		t1 = cv::getTickCount();
		ImageHelper.computePatches(keypoints_1, src_1, patches_1);
		ImageHelper.computeBinaryDescriptors(patches_1, descriptors_1, masks_1);
		t2 = cv::getTickCount();
		tdesc = 1000.0*(t2-t1) / cv::getTickFrequency();

		ImageHelper.computePatches(keypoints_2, src_2, patches_2);
		ImageHelper.computeBinaryDescriptors(patches_2, descriptors_2, masks_2);
	}

	else{
		cout<<argv[2]<<" no es un nombre de descriptor valido"<<endl;
		cout<<"Descriptores: BRIEF,BRISK,FREAK,LDB,LATCH,LATCHK,BAFT,BOLD,(...)"<<endl;
		return 0;
	}

	if ((argc == 5 || argc == 6) && !strcmp (argv[4], "show")){
		//Dibujar kpts en las dos imagenes
		drawKeypoints(src_1, keypoints_1, src_1);
		drawKeypoints(src_2, keypoints_2, src_2);
		imshow("keypoints_1",src_1);
		imshow("keypoints_2",src_2);
		
	}

	//Matching

	cout<<"Matching: "<< descriptors_1.rows<<" descriptores (imagen 1), contra "<< descriptors_2.rows<<
		" descriptores (imagen 2)"<<endl;

	//BFMatcher es comun a BFM y a GMS. En BFM tambien se hace RANSAC. En GMS se aplica un filtro
	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches,good_matches, gms_matches, homography_matches;
	///////////////////////////////

	//Si el matcher es Brute Force Matcher

	if (!strcmp (argv[3], "BFM")){

		vector<vector<DMatch> > matches;     // No es como el matches declarado mas arriba. vector<vector<DMatch> >
		t1 = cv::getTickCount();
		matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);
		t2 = cv::getTickCount();
		tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout << "Tiempo de BFMatcher: " << tmatch << " ms" << endl;


	//Good Matches + RANSAC
		vector<float> distances;

		t1 = cv::getTickCount();
		for(unsigned int i = 0; i < matches.size(); i++ ){
			if (matches[i][0].distance<nn_ratio_threshold *matches[i][1].distance){
				good_matches.push_back(matches[i][0]);
			//cout<< matches[i][0].distance <<" < "<<nn_ratio_threshold *matches[i][1].distance<<endl;
				//cout<<i<<endl;
			}
		}
		cout<<"Good matches: "<<good_matches.size()<<endl;

		vector<Point2f> match_left, match_right;

		for(unsigned int i = 0; i < good_matches.size(); i++ )
		{
			match_left.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
			match_right.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
			distances.push_back(good_matches[i].distance);
			
		}

		Mat correctMatches;
		Mat H = findHomography( match_left, match_right, CV_RANSAC, 3, correctMatches );

	// check if the homography matrix is good enough
		float detMin = 1e-3f;
		if (abs(determinant(H)) < detMin)
			cout<<"Mala Homografia (|det(H)| < "<<detMin<<endl;


		for (unsigned int i=0; i < good_matches.size(); ++i)
		{
			if (*correctMatches.ptr<uchar>(i))
				homography_matches.push_back(good_matches[i]);
		}

		cout<<"Matches correctos (inliers): "<<	homography_matches.size() <<
			" ("<<100.f * (float) homography_matches.size() / (float) good_matches.size()<<"%)"<<endl;

		t2 = cv::getTickCount();
		tmatch += 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo BFM + RANSAC: "<< tmatch <<" ms" <<endl;

	}

	//Si el matcher es GMS

	else if (!strcmp (argv[3], "GMS")){
		t1 = cv::getTickCount();
		matcher.match(descriptors_1, descriptors_2, matches);
		t2 = cv::getTickCount();
		tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<< "Tiempo de BFMatcher: "<< tmatch <<" ms"<<endl;

	//GMS filter
		t1 = cv::getTickCount();
		int num_inliers = 0;
		std::vector<bool> vbInliers;
		gms_matcher gms(keypoints_1,src_1.size(), keypoints_2,src_2.size(), matches);
		num_inliers = gms.GetInlierMask(vbInliers, false, false);

		cout << "Get total " << num_inliers << " matches." << endl;

	//Draw matches
		for (size_t i = 0; i < vbInliers.size(); ++i)
		{
			if (vbInliers[i] == true)
			{
				gms_matches.push_back(matches[i]);
			}
		}

		t2 = cv::getTickCount();
		tmatch += 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo de BFmatcher + filtro GMS: " << tmatch << " ms" << endl;
		cout<<"Good matches: "<<gms_matches.size()<<endl;
		cout<<"Percent: "<<(gms_matches.size()/500.0)*100<<endl;
	}

	//Si el matcher es FLANN
	else if (!strcmp (argv[3], "FLANN")){
		vector<vector<DMatch> > matches;
		//FlannBasedMatcher matcher_flann;

		FlannBasedMatcher matcher_flann(new flann::LshIndexParams(20, 10, 2));
		matcher_flann.knnMatch( descriptors_1, descriptors_2, matches,2 );
		t1 = cv::getTickCount();
		matcher_flann.knnMatch( descriptors_1, descriptors_2, matches,2 );
		//matcher_flann.match( descriptors_1, descriptors_2, matches );
		t2 = cv::getTickCount();
		tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<< "Tiempo de FLANN matcher: "<< tmatch <<" ms"<<endl;

	////////////////////////////////////////////////////////////
	//Calcular good matches: forma 1: como se hizo para BFM   //
	////////////////////////////////////////////////////////////

	//Se deja esta forma de calcular good matches a fines de comparacion con BFM

		for(unsigned int i = 0; i < matches.size(); i++ ){
		if (matches[i][0].distance<nn_ratio_threshold *matches[i][1].distance){
				good_matches.push_back(matches[i][0]);
				
			}
			
		}
		cout<<"Good matches: "<<good_matches.size()<<endl;

	////////////////////////////////////////////////////////////
	//Calcular good matches: forma 2: como venia en el ejemplo//
	////////////////////////////////////////////////////////////


	//-- Quick calculation of max and min distances between keypoints
	/*	t1 = cv::getTickCount();
		double max_dist = 0; double min_dist = 100;
		for( int i = 0; i < descriptors_1.rows; i++ ){
			double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
				if( dist > max_dist ) max_dist = dist;
  		}
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.

		for( int i = 0; i < descriptors_1.rows; i++ ){
			if( matches[i].distance <= max(2*min_dist, 0.02) ){    //Variando 2*min_dist se encuentran mas o menos matches
				good_matches.push_back( matches[i]); 
				//cout << matches[i].distance << " < " << max(10*min_dist, 0.02) << endl;
			}
		}

		t2 = cv::getTickCount();
		tmatch += 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<< "Tiempo de FLANN matcher + Good Matches : "<< tmatch <<" ms"<<endl;

*/


		//RANSAC
		t1 = cv::getTickCount();
		vector<float> distances;
		vector<Point2f> match_left, match_right;

		for(unsigned int i = 0; i < good_matches.size(); i++ )
		{
			match_left.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
			match_right.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
			distances.push_back(good_matches[i].distance);
		}

		Mat correctMatches;
		Mat H = findHomography( match_left, match_right, CV_RANSAC, 3, correctMatches );

	// check if the homography matrix is good enough
		float detMin = 1e-3f;
		if (abs(determinant(H)) < detMin)
			cout<<"Mala Homografia (|det(H)| < "<<detMin<<endl;

		for (unsigned int i=0; i < good_matches.size(); ++i)
		{
			if (*correctMatches.ptr<uchar>(i))
				homography_matches.push_back(good_matches[i]);
		}

		cout<<"Matches correctos (inliers): "<<	homography_matches.size() <<
			" ("<<100.f * (float) homography_matches.size() / (float) good_matches.size()<<"%)"<<endl;

		t2 = cv::getTickCount();
		tmatch += 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Tiempo FLANN + RANSAC: "<< tmatch <<" ms" <<endl;

		/*for( int i = 0; i < (int)good_matches.size(); i++ )
		printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d \n", i, good_matches[i].queryIdx,good_matches[i].trainIdx); 
		*/
	}

	else{
		cout<<argv[3]<<" no es un nombre de matcher valido"<<endl;
		cout<<"Matchers: BFM, GMS, FLANN"<<endl;
		return 0;
	}

	//Mostrar matches

	if ((argc == 5 || argc == 6) && !strcmp (argv[4], "show")){
		// Draw matches
		Mat img_matches;
		if(!strcmp (argv[3], "GMS"))
			drawMatches( src_1, keypoints_1, src_2, keypoints_2, gms_matches, img_matches);
		if(!strcmp (argv[3], "BFM" ) || !strcmp (argv[3], "FLANN" ))
			drawMatches( src_1, keypoints_1, src_2, keypoints_2, homography_matches, img_matches); //Podrian ponerse los good_matches
		/*if(!strcmp (argv[3], "FLANN"))
			drawMatches( src_1, keypoints_1, src_2, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );*/
		imshow("matches",img_matches);
		// Save Image
		imwrite("matches.png", img_matches);
	}

	//Archivo para guardar resultados
	ofstream file("Resultados.txt", ios_base::app);
	//file.open()
	file<<argv[1]<<" + "<<argv[2]<<":"<<endl;
	//file <<"Cantidad de Keypoints: "<< keypoints.size() << endl;
	file <<"Cantidad de Keypoints: "<< kpts << endl;
	file<<"Descriptor size: "<<descriptors_1.size()<<endl;
	file<<"Tiempo deteccion: "<<tdet<<" ms "<<endl;
	file<<"Tiempo descripcion: "<<tdesc<<" ms "<<endl;
	file<<"Tiempo descripcion por keypoint: "<<tdesc*1000/descriptors_1.rows<<" us "<<endl;
	file<<"Tiempo Match: "<<tmatch<<" ms "<<endl;
	file<<"Tiempo total: "<<tdet+tdesc+tmatch<<" ms "<<endl;
	if(!strcmp (argv[3], "GMS"))
		file<<"Good matches: "<<gms_matches.size()<<endl<<endl;
	else if(!strcmp (argv[3], "BFM" ) || !strcmp (argv[3], "FLANN" )){
		file<<"Good matches: "<<good_matches.size()<<endl;
		file<<"Matches correctos (inliers): "<<	homography_matches.size() <<
			" ("<<100.f * (float) homography_matches.size() / (float) good_matches.size()<<"%)"<<endl<<endl;
	}
	//file<<"Matches correctos (inliers): "<<	homography_matches.size() <<
		//" ("<<100.f * (float) homography_matches.size() / (float) good_matches.size()<<"%)"<<endl<<endl;
	waitKey(0);
	return 0;
}

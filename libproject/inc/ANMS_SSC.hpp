#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;

void ssc(vector<cv::KeyPoint>& keyPoints, unsigned int numRetPoints,float tolerance, int cols, int rows){

	//Sorting keypoints by deacreasing order of strength
    if(keyPoints.size() <= 0)
		return;

    vector<int> responseVector;
    for (unsigned int i =0 ; i<keyPoints.size(); i++) responseVector.push_back(keyPoints[i].response);
    vector<int> Indx(responseVector.size()); std::iota (std::begin(Indx), std::end(Indx), 0);
    cv::sortIdx(responseVector, Indx, CV_SORT_DESCENDING);
    vector<cv::KeyPoint> keyPointsSorted;
    for (unsigned int i = 0; i < keyPoints.size(); i++) keyPointsSorted.push_back(keyPoints[Indx[i]]);

	if (numRetPoints > keyPointsSorted.size())
		numRetPoints = keyPointsSorted.size();
    // several temp expression variables to simplify solution equation
    int exp1 = rows + cols + 2*numRetPoints;
    long long exp2 = ((long long) 4*cols + (long long)4*numRetPoints + (long long)4*rows*numRetPoints + (long long)rows*rows + (long long) cols*cols - (long long)2*rows*cols + (long long)4*rows*cols*numRetPoints);
    double exp3 = sqrt(exp2);
    double exp4 = (2*(numRetPoints - 1));

    double sol1 = -round((exp1+exp3)/exp4); // first solution
    double sol2 = -round((exp1-exp3)/exp4); // second solution

    int high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    int low = floor(sqrt((double)keyPointsSorted.size()/numRetPoints));

    int width;
    int prevWidth = -1;

    vector<int> ResultVec;
    bool complete = false;
    unsigned int K = numRetPoints; unsigned int Kmin = round(K-(K*tolerance)); unsigned int Kmax = round(K+(K*tolerance));

    vector<int> result; result.reserve(keyPointsSorted.size());
    while(!complete){
        width = low+(high-low)/2;
        if (width == prevWidth || low>high) { //needed to reassure the same radius is not repeated again
            ResultVec = result; //return the keypoints from the previous iteration
            break;
        }
        result.clear();
        unsigned int c = round(width/2); //initializing Grid
        if (c < 1)
			break;
        int numCellCols = floor(cols/c);
        int numCellRows = floor(rows/c);
        vector<vector<bool> > coveredVec(numCellRows+1,vector<bool>(numCellCols+1,false));

        for (unsigned int i=0;i<keyPointsSorted.size();++i){
            int row = floor(keyPointsSorted[i].pt.y/c); //get position of the cell current point is located at
            int col = floor(keyPointsSorted[i].pt.x/c);
            if (coveredVec[row][col]==false){ // if the cell is not covered
                result.push_back(i);
                int rowMin = ((row-floor(width/c))>=0)? (row-floor(width/c)) : 0; //get range which current radius is covering
                int rowMax = ((row+floor(width/c))<=numCellRows)? (row+floor(width/c)) : numCellRows;
                int colMin = ((col-floor(width/c))>=0)? (col-floor(width/c)) : 0;
                int colMax = ((col+floor(width/c))<=numCellCols)? (col+floor(width/c)) : numCellCols;
                for (int rowToCov=rowMin; rowToCov<=rowMax; ++rowToCov){
                    for (int colToCov=colMin ; colToCov<=colMax; ++colToCov){
                        if (!coveredVec[rowToCov][colToCov]) coveredVec[rowToCov][colToCov] = true; //cover cells within the square bounding box with width w
                    }
                }
            }
        }

        if (result.size()>=Kmin && result.size()<=Kmax){ //solution found
            ResultVec = result;
            complete = true;
        }
        else if (result.size()<Kmin) high = width-1; //update binary search range
        else low = width+1;
        prevWidth = width;
    }
    // retrieve final keypoints
	keyPoints.resize(ResultVec.size());
	keyPoints.clear();
    for (unsigned int i = 0; i<ResultVec.size(); i++) keyPoints.push_back(keyPointsSorted[ResultVec[i]]);

}

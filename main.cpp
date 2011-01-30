#include <iostream>
#include "SudokuUtils.cpp"

using namespace std;
using namespace cv;

int main(int argv, char* argc[])
{
    vector<int> result = scanPuzzle(argc[1]);
    
    for(int i = 0; i < 81; ++i)
    {
        if(i%9 == 0)
            cout << endl;
        if(result[i] != 0)
            cout << result[i] << " ";
        else 
            cout <<"  ";
    }
    return 0;
}


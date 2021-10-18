//关键报错：输入为空时的判断。当rows=0的时候，数组不存在元素，也就不存在matrix[0]，matrix[0]产生越界。

// 二维数组的形参传入：
//vector<vector<int>>& matrix,
// 二维数组的行列求法：
// 行数 int rows=matrix.size();
//列数 int columns=matrix[0].size();
// //总行数为rows，总列数为columns；
// //从右上角开始扫描，（定位出row=0，column=总列数-1）
// //从右上角开始扫描，比目的数字小，则行加一；比目的数字大，列减一。
// //循环，结束的条件是行大于（总行数-1）或列小于0
#include <vector>
#include<iostream>
using namespace std;
    bool findNumberIn2DArray(vector<vector<int> >& matrix, int target){
         bool temp=false;//默认没有找到
         int rows=matrix.size();
                
        if(rows==0)
         {
             return false;
         }
         //再访问matrix[0]就不会越界了
         int columns=matrix[0].size();
         if(columns==0)
         {
             return false;
         }
         cout<<rows<<"    "<<columns;
         if(rows>=0 && columns>=0)
         {
             int row=0;//从小到大
             int column=columns-1;//从大到小
             //进入循环的条件
             while(row<rows&&column>=0)
             {
                 //相等
                 if(matrix[row][column]==target)
                 {
                     temp=true;
                     break;
                 }
                 //从右上角开始扫描，比目的数字小，则行加一
                 else if(matrix[row][column]<target)
                 {
                     ++row;
                 }
                 //从右上角开始扫描，比目的数字大，列减一
                 else 
                 {
                    --column;
                 }

            }
         }
         return temp;

    }
int main(){
    //测试文件
}
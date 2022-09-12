//
// Created by dbettkk on 2022/3/30.
//

#ifndef SPARSECONVOLUTION_CUDATIME_CUH
#define SPARSECONVOLUTION_CUDATIME_CUH

#include<iostream>
#include <string>
#include <fstream>

class CudaTime {
    cudaEvent_t startTime;
    cudaEvent_t endTime;
public:
    void init();
    void start();
    void end();
    float getTime();
    void destroy();

    void initAndStart();

    float endAndGetTime();

    void endAndExportTimeToFile(std::string path, std::string prefix_msg);

    void endAndPrintTime(const std::string &msg);
};


#endif //SPARSECONVOLUTION_CUDATIME_CUH

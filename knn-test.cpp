// knn-test.cpp : Defines the entry point for the console application.
//


#include <algorithm>
#include <execution>
#include <fstream>
#include <iostream>
#include <stdint.h>

/*

http://yann.lecun.com/exdb/mnist/

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

*/


#include "kdtree.h"

std::istream& operator %(std::istream& s, int32_t& v)
{
    s.read((char*)&v, sizeof(v));
    std::reverse((char*)&v, (char*)(&v + 1));
    return s;
}

ObjectInfos ReadDataSet(const char* imageFile, const char* labelFile)
{
    std::ifstream ifsImages(imageFile, std::ifstream::in | std::ifstream::binary);
    int32_t magic;
    ifsImages % magic;
    int32_t numImages;
    ifsImages % numImages;
    int32_t numRows, numCols;
    ifsImages % numRows % numCols;

    std::ifstream ifsLabels(labelFile, std::ifstream::in | std::ifstream::binary);
    ifsLabels % magic;
    int32_t numLabels;
    ifsLabels % numLabels;

    ObjectInfos infos;
    infos.resize(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        ifsImages.read((char*)infos[i].pos, DIM);
        unsigned char label;
        ifsLabels.read((char*)&label, 1);
        infos[i].data = label;
    }

    const bool ok = ifsImages && ifsLabels;

    return infos;
}


int main()
{
    auto trainingSet = ReadDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    std::vector<ObjectInfo*> infoPtrs;
    infoPtrs.reserve(trainingSet.size());

    for (auto it = trainingSet.begin(); it != trainingSet.end(); ++it)
    {
        infoPtrs.push_back(&*it);
    }

    auto root = insert(infoPtrs.begin(), infoPtrs.end(), nullptr, 0);

    auto testSet = ReadDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    std::atomic<int> numMismatches = 0;

    std::for_each(std::execution::par, testSet.begin(), testSet.end(), 
    [root, &numMismatches](const auto& data)
    {
        SearchResults result;
        bool flags[DIM * 2]{};
        kd_nearest_i_nearer_subtree(root, data.pos, result, flags, 0);

        // distance decreases as we go
        const SearchResult* pResult = result.data();
        auto alternative = pResult->data;
        pResult = pResult->next;
        const bool alternativeWins = alternative == pResult->data;
        pResult = pResult->next;
        auto predicted = alternativeWins? alternative : pResult->data;

        if (predicted != data.data)
            ++numMismatches;
    });

    std::cout << "Test cases: " << testSet.size() << "; mismatches: " << numMismatches << '\n';

    return 0;
}

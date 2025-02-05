// knn-test.cpp : Defines the entry point for the console application.
//

#include "nanoflann/include/nanoflann.hpp"

#include <algorithm>
#include <execution>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <time.h>

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

class PixProvider
{
public:
    PixProvider(ObjectInfos& data) : m_data(data) {}
    size_t kdtree_get_point_count() const
    {
        return m_data.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    unsigned char kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return m_data[idx].pos[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

private:
    ObjectInfos& m_data;
};

// construct a kd-tree index:
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<unsigned char, PixProvider, int>,
    PixProvider,
    DIM /* dim */
> my_kd_tree_t;

std::istream& operator %(std::istream& s, int32_t& v)
{
    unsigned char buffer[4];
    s.read((char*)buffer, sizeof(buffer));
    v = (buffer[0] << 24) + (buffer[1] << 16) + (buffer[2] << 8) + buffer[3];
    return s;
}

ObjectInfos ReadDataSet(const char* imageFile, const char* labelFile)
{
    std::string path(__FILE__);
    auto pos = path.find_last_of("\\/");
    if (pos == std::string::npos)
        path.clear();
    else
    {
        path.resize(pos + 1);
        path += "datasets/";
    }

    std::ifstream ifsImages(path + imageFile, std::ifstream::in | std::ifstream::binary);
    int32_t magic;
    ifsImages % magic;
    int32_t numImages;
    ifsImages % numImages;
    int32_t numRows, numCols;
    ifsImages % numRows % numCols;

    std::ifstream ifsLabels(path + labelFile, std::ifstream::in | std::ifstream::binary);
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

    clock_t start = clock();

    std::for_each(//std::execution::par, 
        testSet.begin(), testSet.end(), 
    [root, &numMismatches](const auto& data)
    {
        SearchResults result;
        bool flags[DIM * 2]{};
        DistanceType sq_distances[DIM]{ };
        kd_nearest_i_nearer_subtree<0>(root, data.pos, result, flags, sq_distances);

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

    std::cout << "Test cases: " << testSet.size() << "; mismatches: " << numMismatches
        << "; time: " << (double)(clock() - start) / CLOCKS_PER_SEC <<" seconds\n";

    PixProvider provider(trainingSet);
    my_kd_tree_t infos(DIM, provider);
    //infos.buildIndex();

    numMismatches = 0;
    start = clock();
    std::for_each(//std::execution::par, 
        testSet.begin(), testSet.end(),
        [&trainingSet, &infos, &numMismatches](const auto& data)
        {
            enum { bufSize = 3 };

            uint32_t ret_index[bufSize];
            int out_dist_sqr[bufSize];

            const auto num_results = infos.knnSearch(data.pos, bufSize, &ret_index[0], &out_dist_sqr[0]);

            const auto alternative = trainingSet[ret_index[1]].data;
            const bool alternativeWins = alternative == trainingSet[ret_index[2]].data;
            auto predicted = alternativeWins ? alternative : trainingSet[ret_index[0]].data;

            if (predicted != data.data)
                ++numMismatches;
        });

    std::cout << "Test cases: " << testSet.size() << "; mismatches: " << numMismatches
        << "; nanoflann time: " << (double)(clock() - start) / CLOCKS_PER_SEC << " seconds\n";

    return 0;
}

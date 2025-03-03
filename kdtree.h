﻿#pragma once

#include <algorithm>
#include <vector>


enum { DIM = 28 * 28 };

typedef unsigned char AttributeType;

typedef int DistanceType;

inline DistanceType SQ(DistanceType x) { return x * x; }

struct kdnode
{
    AttributeType pos[DIM];
    int data;
    kdnode *left, *right;
};

typedef kdnode ObjectInfo;


/////////////////////////////////////////////////////////////////////////////////


typedef std::vector<ObjectInfo> ObjectInfos;

struct IsInfoLess
{
    IsInfoLess(int idx) : m_idx(idx) {}

    bool operator() (const ObjectInfo& x, const ObjectInfo& y) const
    {
        return x.pos[m_idx] < y.pos[m_idx];
    }
    bool operator() (const ObjectInfo* x, const ObjectInfo* y) const
    {
        return x->pos[m_idx] < y->pos[m_idx];
    }
private:
    int m_idx;
};

kdnode* insert(std::vector<ObjectInfo*>::iterator begin,
    std::vector<ObjectInfo*>::iterator end,
    kdnode* parent,
    int dir)
{
    if (begin == end)
        return 0;

    int diff = int(end - begin);
    if (1 == diff)
    {
        kdnode* node = *begin;

        node->left = 0;
        node->right = 0;

        return node;
    }

    int halfSize = diff / 2;

    auto middle = begin + halfSize;

    std::nth_element(begin, middle, end, IsInfoLess(dir));

    kdnode* node = *middle;

    const auto new_dir = (dir + 1) % DIM;
    node->left = insert(begin, middle, node, new_dir);
    node->right = insert(++middle, end, node, new_dir);

    return node;
}


struct SearchResult
{
    DistanceType dist_sq;
    SearchResult* next;
    int data;
};

bool operator < (const SearchResult& left, const SearchResult& right)
{
    return left.dist_sq < right.dist_sq;
}

bool operator < (const SearchResult& left, DistanceType right)
{
    return left.dist_sq < right;
}

bool operator < (DistanceType left, const SearchResult& right)
{
    return left < right.dist_sq;
}


class SearchResults
{
public:
    SearchResults()
        : m_pList(0)
        , m_pFree(m_data)
    {
    }

    bool isFull()
    {
        return m_pFree == m_data + 3;
    }
    DistanceType dist_sq()
    {
        return m_pList->dist_sq;
    }
    void insert(DistanceType dist_sq, kdnode* node)
    {
        if (!isFull())
        {
            SearchResult* newResult = m_pFree++;
            newResult->dist_sq = dist_sq;
            newResult->data = node->data;
            doInsert(&m_pList, newResult);
        }
        else if (m_pList->dist_sq > dist_sq)
        {
            SearchResult* newResult = m_pList;
            m_pList = m_pList->next;
            newResult->dist_sq = dist_sq;
            newResult->data = node->data;
            doInsert(&m_pList, newResult);
        }
    }

    const SearchResult* data()
    {
        return m_pList;
    }

    int size()
    {
        return m_pFree - m_data;
    }


private:
    void doInsert(SearchResult** ppResult, SearchResult* toInsert)
    {
        while ((*ppResult) != 0 && (*ppResult)->dist_sq > toInsert->dist_sq)
            ppResult = &((*ppResult)->next);

        toInsert->next = *ppResult;
        *ppResult = toInsert;
    }

    SearchResult m_data[3];

    SearchResult* m_pList;
    SearchResult* m_pFree;
};

template <int dir>
void kd_nearest_i(kdnode *node, const AttributeType *pos,
    SearchResults& result, DistanceType* sq_distances, DistanceType total_distance)
{
    kdnode *nearer_subtree, *farther_subtree;

    /* Decide whether to go left or right in the tree */
    const DistanceType dist = pos[dir] - node->pos[dir];
    if (dist <= 0) {
        nearer_subtree = node->left;
        farther_subtree = node->right;
    }
    else {
        nearer_subtree = node->right;
        farther_subtree = node->left;
    }

    const auto new_dir = (dir + 1) % DIM;

    if (nearer_subtree) {
        /* Recurse down into nearer subtree */
        kd_nearest_i<new_dir>(nearer_subtree, pos, result, sq_distances, total_distance);
    }

    const auto sq_dist = SQ(dist);
    total_distance += sq_dist - sq_distances[dir];
    if (!result.isFull() || total_distance < result.dist_sq())
    {
        /* Check the distance of the point at the current node, compare it with our bests so far */
        //double dist_sq = sq_dist + SQ(node->pos[1 - dir] - pos[1 - dir]);
        if (!result.isFull())
        {
            DistanceType dist_sq = 0;
            for (int i = 0; i < DIM; ++i)
                dist_sq += SQ(node->pos[i] - pos[i]);

            result.insert(dist_sq, node);
        }
        else
        {
            const auto maxDistance = result.dist_sq();
            DistanceType dist_sq = 0;
            int i = 0;
            //for (; i < DIM; ++i)
            //{
            //    dist_sq += SQ(node->pos[i] - pos[i]);
            //    if (dist_sq > maxDistance)
            //        break;
            //}

            switch (DIM % 8) {
            case 0: do { dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 7:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 6:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 5:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 4:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 3:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 2:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
            case 1:      dist_sq += SQ(node->pos[i] - pos[i]); ++i;
                if (dist_sq > maxDistance)
                    goto too_far;
            } while (i < DIM);
            }

            //if (i == DIM)
                result.insert(dist_sq, node);
            too_far:;
        }

        if (farther_subtree)
        {
            auto save_dist = sq_distances[dir];
            sq_distances[dir] = sq_dist;

            /* Recurse down into farther subtree */
            kd_nearest_i<new_dir>(farther_subtree, pos, result, sq_distances, total_distance);

            sq_distances[dir] = save_dist;
        }
    }
}

template <int dir>
void kd_nearest_i_nearer_subtree(kdnode *node, const AttributeType *pos,
    SearchResults& result, bool* flags, DistanceType* sq_distances)
{
    kdnode *nearer_subtree, *farther_subtree;
    int flagIdx;

    /* Decide whether to go left or right in the tree */
    const DistanceType dist = pos[dir] - node->pos[dir];
    if (dist <= 0) {
        nearer_subtree = node->left;
        farther_subtree = node->right;
        flagIdx = dir * 2;
    }
    else {
        nearer_subtree = node->right;
        farther_subtree = node->left;
        flagIdx = dir * 2 + 1;
    }

    const auto new_dir = (dir + 1) % DIM;

    if (nearer_subtree) {
        /* Recurse down into nearer subtree */
        kd_nearest_i_nearer_subtree<new_dir>(nearer_subtree, pos, result, flags, sq_distances);
    }

    if (flags[flagIdx])
        return;

    const auto sq_dist = SQ(dist);
    if (!result.isFull() || sq_dist < result.dist_sq())
    {
        if (node->pos != pos)
        {
            /* Check the distance of the point at the current node, compare it with our bests so far */
            //double dist_sq = sq_dist + SQ(node->pos[1 - dir] - pos[1 - dir]);
            DistanceType dist_sq = 0;
            for (int i = 0; i < DIM; ++i)
                dist_sq += SQ(node->pos[i] - pos[i]);

            result.insert(dist_sq, node);
        }

        if (farther_subtree)
        {
            //DistanceType sq_distances[DIM]{ };
            //sq_distances[1 - dir] = 0;
            sq_distances[dir] = sq_dist;

            /* Recurse down into farther subtree */
            kd_nearest_i<new_dir>(farther_subtree, pos, result, sq_distances, sq_dist);
            sq_distances[dir] = 0;
        }
    }
    else
        flags[flagIdx] = true;
}

#pragma once
//#include <bits/stdc++.h>
#include <vector>
#include <memory> // for shared_ptr
#include <algorithm> // for std::max
#include <iostream>

class Mem_pool {
private:
    size_t m_capacity;
    void *mem_ptr;
    std::vector<size_t> frames;
    std::vector<size_t> free_set;

public:
    Mem_pool() = delete;
    Mem_pool(const Mem_pool &pool) = delete;
    Mem_pool(size_t _size);
    void *allocate(size_t size);
    void free(void *ptr);
    size_t size() const;
    size_t capacity() const;
    ~Mem_pool();
};

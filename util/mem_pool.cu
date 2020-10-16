#include "mem_pool.h"
#include <cuda_runtime_api.h>
#define cudaChk(stat)                                                                              \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
static void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
        exit(1);
    }
}

Mem_pool::Mem_pool(size_t _size) : m_capacity(_size) {
    cudaChk(cudaMalloc((void **)&mem_ptr, m_capacity));
    frames.emplace_back(reinterpret_cast<size_t>(mem_ptr));
}

void *Mem_pool::allocate(size_t size) {
    const size_t addr = frames.back();
    if (addr + size > reinterpret_cast<size_t>(mem_ptr) + m_capacity) {
        std::cout << "GPU out of memory. Create frame failed\n";
        exit(1);
    }
    frames.emplace_back(addr + size);
    return reinterpret_cast<void *>(addr);
};

void Mem_pool::free(void *ptr) {
    auto frontier = reinterpret_cast<void *>(frames[frames.size() - 2]);
    if (ptr == frontier) {
        frames.pop_back();
        for (int i = 0; i < free_set.size(); i++) {
            if (frames[frames.size() - 2] == free_set[i]) {
                frames.pop_back();
                auto iter = free_set.begin() + i;
                free_set.erase(iter);
                i = -1;
            }
        }
    } else {
        free_set.push_back(reinterpret_cast<size_t>(ptr));
    }
}

Mem_pool::~Mem_pool() { cudaChk(cudaFree(mem_ptr)); }

size_t Mem_pool::size() const {
    size_t b_addr = reinterpret_cast<size_t>(frames.front());
    size_t f_addr = reinterpret_cast<size_t>(frames.back());
    return f_addr - b_addr;
}

size_t Mem_pool::capacity() const { return m_capacity; }

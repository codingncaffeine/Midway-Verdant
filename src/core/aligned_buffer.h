#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>

#include "core/types.h"

namespace mv {

// Host memory that a hypervisor may map: page-aligned (2 MiB, so the host can back it with large pages) and
// zero-filled. Movable, not copyable.
class AlignedBuffer {
public:
    static constexpr std::size_t kAlignment = 2u * 1024 * 1024;

    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t size, u8 fill = 0) { allocate(size, fill); }
    ~AlignedBuffer() { release(); }
    AlignedBuffer(AlignedBuffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void allocate(std::size_t size, u8 fill = 0) {
        release();
        // Round the size up to the alignment: aligned_alloc wants a multiple on some platforms.
        const std::size_t rounded = (size + kAlignment - 1) / kAlignment * kAlignment;
#if defined(_MSC_VER)
        data_ = static_cast<u8*>(_aligned_malloc(rounded, kAlignment));
#else
        data_ = static_cast<u8*>(std::aligned_alloc(kAlignment, rounded));
#endif
        if (!data_) throw std::bad_alloc{};
        std::memset(data_, fill, rounded);
        size_ = size;
    }

    u8* data() { return data_; }
    const u8* data() const { return data_; }
    std::size_t size() const { return size_; }
    u8& operator[](std::size_t i) { return data_[i]; }
    const u8& operator[](std::size_t i) const { return data_[i]; }

private:
    void release() {
        if (!data_) return;
#if defined(_MSC_VER)
        _aligned_free(data_);
#else
        std::free(data_);
#endif
        data_ = nullptr;
        size_ = 0;
    }
    u8* data_{nullptr};
    std::size_t size_{0};
};

}  // namespace mv

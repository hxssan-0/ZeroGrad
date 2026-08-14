#include <zerograd/arena.hpp>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>

namespace zerograd
{
    Arena::Arena(std::size_t total_bytes)
    {
        std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        buffer_length = ((total_bytes + page_size - 1) / page_size) * page_size;

        void* mapped = mmap(
            nullptr, buffer_length + page_size, PROT_READ | PROT_WRITE, 
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
        );

        if (mapped == MAP_FAILED) {
            throw std::runtime_error("Arena: mmap failed");
        }

        if (mprotect(static_cast<uint8_t*>(mapped) + buffer_length, page_size, PROT_NONE) != 0) {
            munmap(mapped, buffer_length + page_size);
            throw std::runtime_error("Arena: mprotect failed");
        }

        buffer = static_cast<uint8_t*>(mapped);
    }

    Arena::~Arena()
    {
        if (buffer) {
            std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
            munmap(buffer, buffer_length + page_size);
        }
    }

    void Arena::reset()
    {
        std::memset(buffer, 0, buffer_length);
        cursor = 0;
    }

    Arena::Arena(Arena&& other) noexcept : buffer(other.buffer), buffer_length(other.buffer_length)
    {
        other.buffer = nullptr;
        other.buffer_length = 0;
    }

    Arena& Arena::operator=(Arena&& other) noexcept
    {
        if (this != &other) {
            if (buffer) {
                std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
                munmap(buffer, buffer_length + page_size);
            }

            buffer = other.buffer;
            buffer_length = other.buffer_length;

            other.buffer = nullptr;
            other.buffer_length = 0;
        }

        return *this;
    }

    void* Arena::alloc(std::size_t size, std::size_t align)
    {
        std::size_t current = reinterpret_cast<std::size_t>(buffer) + cursor;
        std::size_t aligned = (current + align - 1) & ~(align - 1);
        std::size_t new_cursor = (aligned - reinterpret_cast<std::size_t>(buffer)) + size;

        if (new_cursor > buffer_length) {
            throw std::out_of_range("Arena::alloc: out of capacity");
        }

        cursor = new_cursor;
        return reinterpret_cast<void*>(aligned);
    }
}
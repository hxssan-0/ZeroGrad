#pragma once

#include <cstdint>
#include <stdexcept>

namespace zerograd
{
    class Arena
    {
    private:
        uint8_t* buffer = nullptr;
        std::size_t buffer_length = 0;
        std::size_t cursor = 0;
    
    public:
        explicit Arena(std::size_t total_bytes);
        ~Arena();

        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        Arena(Arena&&) noexcept;
        Arena& operator=(Arena&&) noexcept;

        void* alloc(std::size_t size, std::size_t align = alignof(std::max_align_t));

        template <typename T>
        T* get_ptr(std::size_t offset_bytes) const
        {
            if (offset_bytes + sizeof(T) > buffer_length) {
                throw std::out_of_range("Arena::get_ptr: offset + size exceeds arena capacity");
            }

            return reinterpret_cast<T*>(buffer + offset_bytes);
        }

        void reset();
    };

    class ScopedArena
    {
    private:
        Arena& arena;

    public:
        explicit ScopedArena(Arena& arena_) : arena(arena_) {}
        ~ScopedArena() { arena.reset(); }

        ScopedArena(const ScopedArena&) = delete;
        ScopedArena operator=(const ScopedArena&) = delete;
    };
}
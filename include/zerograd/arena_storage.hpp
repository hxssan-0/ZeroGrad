#pragma once

#include <vector>
#include <utility>
#include <span>
#include <cstddef>
#include <algorithm>
#include <initializer_list>

namespace zerograd
{
    template <typename T>
    class ArenaStorage
    {
    private:
        std::vector<T> owned;
        T* view = nullptr;
        std::size_t count = 0;

    public:
        explicit ArenaStorage(std::vector<T> owned_) : owned(std::move(owned_)), count(owned.size()) {}
        ArenaStorage(T* arena_ptr, std::size_t count_) : view(arena_ptr), count(count_) {}

        T& operator[](std::size_t i) { return view ? view[i] : owned[i]; }
        const T& operator[](std::size_t i) const { return view ? view[i] : owned[i]; }
        std::size_t size() const { return count; }
        T* begin() { return view ? view : owned.data(); }
        T* end() { return begin() + count; }
        const T* begin() const { return view ? view : owned.data(); }
        const T* end() const { return begin() + count; }

        operator std::vector<T>() const { return std::vector<T>(begin(), end()); }
        operator std::span<const T>() const { return std::span<const T>(begin(), count); }

        bool operator==(const std::vector<T>& other) const {
            if (this->size() != other.size()) return false;
            return std::equal(this->begin(), this->end(), other.begin());
        }

        friend bool operator==(const std::vector<T>& lhs, const ArenaStorage<T>& rhs) {
            return rhs == lhs;
        }

        ArenaStorage& operator=(std::initializer_list<T> ilist) {
            std::copy(ilist.begin(), ilist.end(), this->begin());
            return *this;
        }
    };

    using TensorStorage = ArenaStorage<float>;
    using ShapeStorage = ArenaStorage<std::size_t>;
}
#include "tensor.hpp"

namespace zerograd
{
    class Arena
    {
    private:
        uint8_t* buffer = nullptr;
        std::size_t buffer_length = 0;
    
    public:
        explicit Arena(std::size_t total_bytes);
        ~Arena();

        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        Arena(Arena&&) noexcept;
        Arena& operator=(Arena&&) noexcept;

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
}
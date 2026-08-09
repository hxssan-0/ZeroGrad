namespace zerograd
{
    inline std::size_t& global_step_counter()
    {
        static std::size_t step_id = 0;
        return step_id;
    }
}
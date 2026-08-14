#include <catch2/catch_test_macros.hpp>
#include <zerograd/arena.hpp>
#include <random>
#include <unistd.h>

TEST_CASE("Arena: 1000 objects of varying sizes do not overlap", "[arena]") {
    zerograd::Arena arena(1'000'000);

    std::mt19937 gen(123);
    std::uniform_int_distribution<std::size_t> size_dist(1, 512);

    std::vector<std::pair<std::size_t, std::size_t>> placed;
    std::size_t cursor = 0;

    for (int i = 0; i < 1000; ++i) {
        std::size_t sz = size_dist(gen);
        REQUIRE(cursor + sz <= 1'000'000);

        placed.emplace_back(cursor, sz);
        cursor += sz;
    }

    for (std::size_t i = 0; i < placed.size(); ++i) {
        for (std::size_t j = i + 1; j < placed.size(); ++j) {
            auto [off_i, sz_i] = placed[i];
            auto [off_j, sz_j] = placed[j];
            bool overlap = (off_i < off_j + sz_j) && (off_j < off_i + sz_i);
            REQUIRE_FALSE(overlap);
        }
    }

    for (auto& [offset, size] : placed) {
        uint8_t* p = arena.get_ptr<uint8_t>(offset);
        std::memset(p, 0xAB, size);
    }
}

TEST_CASE("Arena: alignment is correct", "[arena]") {
    zerograd::Arena arena(1'000'000);

    long page_size = sysconf(_SC_PAGESIZE);
    auto* base_ptr = arena.get_ptr<uint8_t>(0);

    REQUIRE(reinterpret_cast<std::uintptr_t>(base_ptr) % static_cast<std::uintptr_t>(page_size) == 0);
}

TEST_CASE("Arena: reset zeroes previously written memory", "[arena]") {
    zerograd::Arena arena(4096);

    uint8_t* p = arena.get_ptr<uint8_t>(0);
    std::memset(p, 0xFF, 100);
    REQUIRE(p[0] == 0xFF);
    REQUIRE(p[99] == 0xFF);

    arena.reset();

    REQUIRE(p[0] == 0);
    REQUIRE(p[99] == 0);
}

TEST_CASE("Arena: move leaves source safely destructible", "[arena]") {
    zerograd::Arena a(4096);
    uint8_t* original_ptr = a.get_ptr<uint8_t>(0);

    zerograd::Arena b(std::move(a));

    b.get_ptr<uint8_t>(0)[0] = 42;
    REQUIRE(b.get_ptr<uint8_t>(0)[0] == 42);
}

TEST_CASE("Arena: stress test with 10 million allocations", "[arena]") {
    std::size_t total_bytes = 10'000'000ULL * 64;
    zerograd::Arena arena(total_bytes);

    std::size_t cursor = 0;
    for (std::size_t i = 0; i < 10'000'000; ++i) {
        std::size_t sz = 64;
        REQUIRE(cursor + sz <= total_bytes);
        uint8_t* p = arena.get_ptr<uint8_t>(cursor);
        p[0] = static_cast<uint8_t>(i & 0xFF);
        cursor += sz;
    }
}
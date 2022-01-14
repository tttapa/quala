#include <gtest/gtest.h>
#include <quala/util/ringbuffer.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

TEST(MaxHistory, updown) {
    quala::MaxHistory<int> m(3);
    m.add(1);
    EXPECT_EQ(m.max(), 1);
    m.add(2);
    EXPECT_EQ(m.max(), 2);
    m.add(3);
    EXPECT_EQ(m.max(), 3);
    m.add(4);
    EXPECT_EQ(m.max(), 4);
    m.add(5);
    EXPECT_EQ(m.max(), 5);
    m.add(5);
    EXPECT_EQ(m.max(), 5);
    m.add(6);
    EXPECT_EQ(m.max(), 6);
    m.add(7);
    EXPECT_EQ(m.max(), 7);
    m.add(8);
    EXPECT_EQ(m.max(), 8);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(8);
    EXPECT_EQ(m.max(), 9);
    m.add(8);
    EXPECT_EQ(m.max(), 9);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(8);
    EXPECT_EQ(m.max(), 9);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(8);
    EXPECT_EQ(m.max(), 9);
    m.add(7);
    EXPECT_EQ(m.max(), 9);
    m.add(6);
    EXPECT_EQ(m.max(), 8);
    m.add(5);
    EXPECT_EQ(m.max(), 7);
    m.add(4);
    EXPECT_EQ(m.max(), 6);
    m.add(3);
    EXPECT_EQ(m.max(), 5);
}

TEST(MaxHistory, updown2) {
    quala::MaxHistory<int> m(999);
    m.add(1);
    EXPECT_EQ(m.max(), 1);
    m.add(2);
    EXPECT_EQ(m.max(), 2);
    m.add(3);
    EXPECT_EQ(m.max(), 3);
    m.add(4);
    EXPECT_EQ(m.max(), 4);
    m.add(5);
    EXPECT_EQ(m.max(), 5);
    m.add(5);
    EXPECT_EQ(m.max(), 5);
    m.add(6);
    EXPECT_EQ(m.max(), 6);
    m.add(7);
    EXPECT_EQ(m.max(), 7);
    m.add(8);
    EXPECT_EQ(m.max(), 8);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 10);
    m.add(8);
    EXPECT_EQ(m.max(), 10);
    m.add(100);
    EXPECT_EQ(m.max(), 100);
}

TEST(MaxHistory, downup) {
    quala::MaxHistory<int> m(3);
    m.add(10);
    EXPECT_EQ(m.max(), 10);
    m.add(9);
    EXPECT_EQ(m.max(), 10);
    m.add(8);
    EXPECT_EQ(m.max(), 10);
    m.add(7);
    EXPECT_EQ(m.max(), 9);
    m.add(6);
    EXPECT_EQ(m.max(), 8);
    m.add(5);
    EXPECT_EQ(m.max(), 7);
    m.add(4);
    EXPECT_EQ(m.max(), 6);
    m.add(7);
    EXPECT_EQ(m.max(), 7);
    m.add(8);
    EXPECT_EQ(m.max(), 8);
    m.add(9);
    EXPECT_EQ(m.max(), 9);
}

auto circular  = [](quala::CircularIndices<int> i) { return i.circular; };
auto zerobased = [](quala::CircularIndices<int> i) { return i.zerobased; };

TEST(CircularRange, mutforward) {
    quala::CircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(CircularRange, constforward) {
    const quala::CircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(CircularRange, forwardfull) {
    quala::CircularRange<int> r{5, 3, 3, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3, 4}, expected_c{3, 4, 0, 1, 2};
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(CircularRange, mutreverse) {
    quala::CircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    std::reverse(expected_z.begin(), expected_z.end());
    std::reverse(expected_c.begin(), expected_c.end());
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(CircularRange, constreverse) {
    const quala::CircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    std::reverse(expected_z.begin(), expected_z.end());
    std::reverse(expected_c.begin(), expected_c.end());
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(CircularRange, reversefull) {
    quala::CircularRange<int> r{5, 3, 3, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3, 4}, expected_c{3, 4, 0, 1, 2};
    std::reverse(expected_z.begin(), expected_z.end());
    std::reverse(expected_c.begin(), expected_c.end());
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(CircularRange, iteratortraits) {
    quala::CircularRange<int> r{5, 3, 3, 5};
    auto a  = std::next(r.begin());
    using I = decltype(a);

    // // https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator
    // EXPECT_EQ(--(++a), a);
    // EXPECT_EQ(&(--a), &a);
    // EXPECT_TRUE((std::is_same_v<decltype(--a), I &>));
    // EXPECT_TRUE((std::is_convertible_v<decltype(a--), const I &>));
    // EXPECT_TRUE(
    //     (std::is_same_v<decltype(*a--), std::iterator_traits<I>::reference>));

    // // https://en.cppreference.com/w/cpp/named_req/ForwardIterator
    // EXPECT_EQ([](I a) { return a++; }(a),
    //           [](I a) {
    //               I ip = a;
    //               ++a;
    //               return ip;
    //           }(a));
    // EXPECT_TRUE(
    //     (std::is_lvalue_reference_v<std::iterator_traits<I>::reference>));
    // EXPECT_TRUE((std::is_same_v<std::remove_cv_t<std::remove_reference_t<
    //                                 std::iterator_traits<I>::reference>>,
    //                             std::iterator_traits<I>::value_type>));
    // EXPECT_TRUE((std::is_convertible_v<decltype(a++), const I &>));
    // EXPECT_TRUE(
    //     (std::is_same_v<decltype(*a++), std::iterator_traits<I>::reference>));
    // EXPECT_TRUE((std::is_default_constructible_v<I>));
    // // Note: multipass guarantee is not satisfied because a == b does not imply
    // //       that the references *a and *b are bound to the same object.

    // https://en.cppreference.com/w/cpp/named_req/InputIterator
    auto b = a;
    EXPECT_EQ(a, b);
    EXPECT_EQ(a != b, !(a == b));
    b = std::next(b);
    EXPECT_NE(a, b);
    EXPECT_EQ(a != b, !(a == b));
    // EXPECT_EQ(&(a->zerobased), &((*a).zerobased));
    EXPECT_EQ([](I a) { return *a++; }(a),
              [](I a) {
                  std::iterator_traits<I>::value_type x = *a;
                  ++a;
                  return x;
              }(a));
    EXPECT_TRUE(std::is_signed_v<std::iterator_traits<I>::difference_type>);

    // https://en.cppreference.com/w/cpp/named_req/Iterator
    EXPECT_TRUE(std::is_copy_constructible_v<I>);
    EXPECT_TRUE(std::is_copy_assignable_v<I>);
    EXPECT_TRUE(std::is_destructible_v<I>);
    EXPECT_TRUE(std::is_swappable_v<I>);
    EXPECT_TRUE((std::is_same_v<decltype(++a), I &>));
}

TEST(ReverseCircularRange, mutforward) {
    quala::ReverseCircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    std::reverse(expected_z.begin(), expected_z.end());
    std::reverse(expected_c.begin(), expected_c.end());
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(ReverseCircularRange, constforward) {
    const quala::ReverseCircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    std::reverse(expected_z.begin(), expected_z.end());
    std::reverse(expected_c.begin(), expected_c.end());
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(ReverseCircularRange, forwardfull) {
    quala::ReverseCircularRange<int> r{5, 3, 3, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.begin(), r.end(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3, 4}, expected_c{3, 4, 0, 1, 2};
    std::reverse(expected_z.begin(), expected_z.end());
    std::reverse(expected_c.begin(), expected_c.end());
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(ReverseCircularRange, mutreverse) {
    quala::ReverseCircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(ReverseCircularRange, constreverse) {
    const quala::ReverseCircularRange<int> r{4, 3, 2, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3}, expected_c{3, 4, 0, 1};
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}

TEST(ReverseCircularRange, reversefull) {
    quala::ReverseCircularRange<int> r{5, 3, 3, 5};
    std::vector<int> result_z, result_c;
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_z),
                   zerobased);
    std::transform(r.rbegin(), r.rend(), std::back_insert_iterator(result_c),
                   circular);
    std::vector<int> expected_z{0, 1, 2, 3, 4}, expected_c{3, 4, 0, 1, 2};
    EXPECT_EQ(result_z, expected_z);
    EXPECT_EQ(result_c, expected_c);
}
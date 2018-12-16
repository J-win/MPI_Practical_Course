//  Copyright: (c) lesya89
#ifndef MODULES_TASK_3_ZOLOTAREVA_TASK3_LSD_RADIX_SORT_INCLUDE_RADIX_SORT_H_
#define MODULES_TASK_3_ZOLOTAREVA_TASK3_LSD_RADIX_SORT_INCLUDE_RADIX_SORT_H_

#include <algorithm>  // For swap
#include <cstdint>    // For system independent base types
#include <cstring>    // For memset
#include <vector>
#include <utility>

template <typename ValueType>
void RadixSortMSD(ValueType* array, size_t N);

namespace details {

using Radixable8 = uint8_t;
using Radixable16 = uint16_t;
using Radixable32 = uint32_t;
using Radixable64 = uint64_t;

template <size_t Bytes>
struct Radixable {};

template <>
struct Radixable<1> {
  using type = Radixable8;
};
template <>
struct Radixable<2> {
  using type = Radixable16;
};
template <>
struct Radixable<4> {
  using type = Radixable32;
};
template <>
struct Radixable<8> {
  using type = Radixable64;
};

template <class ValueType, class RadixableType>
struct RadixTransform {
  RadixableType TransformFrom(ValueType);
  ValueType TransformTo(RadixableType);
};

/*
 * The first group of transformation rules are by far the easiest and most
 * straight forward. Unsigned numbers are already in radixable form.
 */

template <>
struct RadixTransform<double, Radixable64> {
  Radixable64 TransformFrom(double value) {
    union _ {double value; Radixable64 cg;};
    Radixable64 rdxble = ((union _ *)&value)->cg;
    // Two's complement negative numbers, but keep them negative.
    if ((rdxble >> 63) == 1) {
      rdxble *= -1;
      rdxble ^= (((Radixable64)1) << 63);
    }
    // Flip the high bit on all numbers to swap the positive and negative
    // regions.
    rdxble ^= (((Radixable64)1) << 63);
    return rdxble;
  }

  double TransformTo(Radixable64 rdxble) {
    rdxble ^= (((Radixable64)1) << 63);
    if ((rdxble >> 63) == 1) {
      rdxble ^= (((Radixable64)1) << 63);
      rdxble *= -1;
    }
    union __ { double fg; Radixable64 rdxble;};
    double rdxdbl = ((union __ *)&rdxble)->fg;
    return rdxdbl;
  }
};


template <typename RadixableType>
void RadixSortMSD(RadixableType* data, size_t N, size_t bit) {
  if (N <= 1 || std::is_sorted(data, data + N)) {
    return;
  }

  int left = 0;
  int right = N;
  while (left < right) {
    size_t left_bit = ((data[left] >> bit) & 1);

    if (left_bit == 0) {
      ++left;
      continue;
    }

    size_t right_bit = ((data[right - 1] >> bit) & 1);
    if (right_bit == 1) {
      --right;
      continue;
    }

    std::swap(data[left], data[right - 1]);
    ++left;
    --right;
  }

  /*
   * As long as more bits exist, call both sides of the pivot recursively.
   */
  if (bit > 0) {
    RadixSortMSD(data, left, bit - 1);
    RadixSortMSD(data + left, N - left, bit - 1);
  }
}

template <typename RadixableType, typename ValueType>
void TransformAndInvoke(ValueType* array, size_t N,
                        void (*RadixFunc)(RadixableType*, size_t, size_t)) {
  RadixTransform<ValueType, RadixableType> transformer;

  RadixableType* radixable_array = reinterpret_cast<RadixableType*>(array);

  for (size_t i = 0; i < N; ++i) {
    radixable_array[i] = transformer.TransformFrom(array[i]);
  }

  RadixFunc(radixable_array, N, (8 * sizeof(RadixableType)) - 1);

  for (size_t i = 0; i < N; ++i) {
    array[i] = transformer.TransformTo(radixable_array[i]);
  }
}

}  // namespace details

/* Actual implementation of MSD Radix Sort. */
template <typename ValueType>
void RadixSortMSD(ValueType* array, size_t N) {
  using RadixableType = typename details::Radixable<sizeof(ValueType)>::type;
  details::TransformAndInvoke<RadixableType>(array, N, details::RadixSortMSD);
}

#endif  // MODULES_TASK_3_ZOLOTAREVA_TASK3_LSD_RADIX_SORT_INCLUDE_RADIX_SORT_H_

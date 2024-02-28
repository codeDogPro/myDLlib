#pragma once

#include <immintrin.h>

namespace dl{

  float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);           // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
  }

  float hsum256_ps_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);       // add the low 128
    return hsum_ps_sse3( vlow); // and inline the sse3 version, which is optimal for AVX // (n
  }

} // namespace dl

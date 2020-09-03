// -----------------------------------------------------------------------------------------
// edgelevel by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2020 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include <Windows.h>
#include <algorithm>
#include <immintrin.h>
#include <process.h>
#include <stdint.h>
#pragma comment(lib, "winmm.lib")
#include "edgelevel.h"

//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#define _mm256_bsrli_epi128 _mm256_srli_si256
#define _mm256_bslli_epi128 _mm256_slli_si256

//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))

#define _mm256_store_switch_si256(ptr, xmm) ((aligned_store) ? _mm256_stream_si256((__m256i *)(ptr), (xmm)) : _mm256_storeu_si256((__m256i *)(ptr), (xmm)))

static inline __m256i _mm256_cmpgt_epu8(__m256i x, __m256i y) {
    // Returns 0xFF where x > y:
    return _mm256_andnot_si256(
        _mm256_cmpeq_epi8(x, y),
        _mm256_cmpeq_epi8(_mm256_max_epu8(x, y), x)
    );
}
static inline __m256i _mm256_cmple_epu16(__m256i x, __m256i y) {
    // Returns 0xFFFF where x <= y:
    return _mm256_cmpeq_epi16(_mm256_subs_epu16(x, y), _mm256_setzero_si256());
}
static inline __m256i _mm256_cmpgt_epu16(__m256i x, __m256i y) {
    // Returns 0xFFFF where x > y:
    return _mm256_andnot_si256(_mm256_cmpeq_epi16(x, y), _mm256_cmple_epu16(y, x));
}
static __forceinline __m256i get_previous_2_y_pixels_256(const char *src) {
    return _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src - 16)), _mm_setzero_si128());
}

static __forceinline __m256i edgelevel_avx2_8(
    const __m256i &ySrc1YM2, const __m256i &ySrc1YM1,
    const __m256i &ySrc0, const __m256i &ySrc1, const __m256i &ySrc2,
    const __m256i &ySrc1YP1, const __m256i &ySrc1YP2,
    const __m256i &yThreshold, const __m256i &yStrength, const __m256i &yWc, const __m256i &yBc) {

    __m256i xVmax, xVmin;
    xVmax = _mm256_max_epu8(ySrc1, ySrc1YM2);
    xVmin = _mm256_min_epu8(ySrc1, ySrc1YM2);
    xVmax = _mm256_max_epu8(xVmax, ySrc1YM1);
    xVmin = _mm256_min_epu8(xVmin, ySrc1YM1);
    xVmax = _mm256_max_epu8(xVmax, ySrc1YP1);
    xVmin = _mm256_min_epu8(xVmin, ySrc1YP1);
    xVmax = _mm256_max_epu8(xVmax, ySrc1YP2);
    xVmin = _mm256_min_epu8(xVmin, ySrc1YP2);

    __m256i ySrc1XM2 = _mm256_alignr256_epi8(ySrc1, ySrc0, 32 - 2);
    __m256i ySrc1XM1 = _mm256_alignr256_epi8(ySrc1, ySrc0, 32 - 1);
    __m256i ySrc1XP1 = _mm256_alignr256_epi8(ySrc2, ySrc1, 1);
    __m256i ySrc1XP2 = _mm256_alignr256_epi8(ySrc2, ySrc1, 2);
    __m256i xMax, xMin;
    xMax = _mm256_max_epu8(ySrc1, ySrc1XM2);
    xMin = _mm256_min_epu8(ySrc1, ySrc1XM2);
    xMax = _mm256_max_epu8(xMax, ySrc1XM1);
    xMin = _mm256_min_epu8(xMin, ySrc1XM1);
    xMax = _mm256_max_epu8(xMax, ySrc1XP1);
    xMin = _mm256_min_epu8(xMin, ySrc1XP1);
    xMax = _mm256_max_epu8(xMax, ySrc1XP2);
    xMin = _mm256_min_epu8(xMin, ySrc1XP2);

    //if (max - min < vmax - vmin) { max = vmax; min = vmin; }
    __m256i xMask = _mm256_cmpgt_epu8(_mm256_sub_epi8(xVmax, xVmin), _mm256_sub_epi8(xMax, xMin));
    xMax  = _mm256_blendv_epi8(xMax, xVmax, xMask);
    xMin  = _mm256_blendv_epi8(xMin, xVmin, xMask);

    //avg = (min + max + 1) >> 1;
    __m256i xAvg  = _mm256_avg_epu8(xMax, xMin);

    //if (max - min > thrs)
    xMask = _mm256_cmpgt_epu8(_mm256_subs_epi8(xMax, xMin), yThreshold);

    //if (src->y == max) max += wc * 2;
    //else max += wc;
    __m256i xMaskMax = _mm256_cmpeq_epi8(ySrc1, xMax);
    xMax  = _mm256_adds_epu8(xMax, yWc);
    xMax  = _mm256_adds_epu8(xMax, _mm256_and_si256(yWc, xMaskMax));

    //if (src->y == min) min -= bc * 2;
    //else  min -= bc;
    __m256i xMaskMin = _mm256_cmpeq_epi8(ySrc1, xMin);
    xMin  = _mm256_subs_epu8(xMin, yBc);
    xMin  = _mm256_subs_epu8(xMin, _mm256_and_si256(yBc, xMaskMin));

    //dst[x] = (BYTE)clamp(src[x] + (((src[x] - avg) * strength) >> 4), (std::max)(min, 0), (std::min)(max, 255));
    __m256i y0, y1;
    y0    = _mm256_sub_epi16(_mm256_unpacklo_epi8(ySrc1, _mm256_setzero_si256()), _mm256_unpacklo_epi8(xAvg, _mm256_setzero_si256()));
    y1    = _mm256_sub_epi16(_mm256_unpackhi_epi8(ySrc1, _mm256_setzero_si256()), _mm256_unpackhi_epi8(xAvg, _mm256_setzero_si256()));
    y0    = _mm256_mullo_epi16(y0, yStrength);
    y1    = _mm256_mullo_epi16(y1, yStrength);
    y0    = _mm256_srai_epi16(y0, 4);
    y1    = _mm256_srai_epi16(y1, 4);
    y0    = _mm256_add_epi16(y0, _mm256_unpacklo_epi8(ySrc1, _mm256_setzero_si256()));
    y1    = _mm256_add_epi16(y1, _mm256_unpackhi_epi8(ySrc1, _mm256_setzero_si256()));
    y0    = _mm256_packus_epi16(y0, y1);
    y0    = _mm256_min_epu8(y0, xMax);
    y0    = _mm256_max_epu8(y0, xMin);

    return _mm256_blendv_epi8(ySrc1, y0, xMask);
}

static __forceinline void edgelevel_avx2_8_line(
    char *dst_line, const int dst_pitch,
    const char *src_line, const int src_pitch, int w,
    const __m256i &yThreshold, const __m256i &yStrength, const __m256i &yWc, const __m256i &yBc) {
    const char *src = src_line;
    char *dst = dst_line;
    __m256i ySrc0 = get_previous_2_y_pixels_256(src);
    __m256i ySrc1 = _mm256_loadu_si256((const __m256i *)(src));
    __m256i ySrc1YM2, ySrc1YM1, ySrc1YP1, ySrc1YP2, ySrc2;
    const int loopsize = 32;
    const char *src_fin = src + (w & (~(loopsize - 1))) * sizeof(uint16_t);
    for (; src < src_fin; src += 32, dst += 32) {
        //周辺近傍の最大と最小を縦方向・横方向に求める
        ySrc1YM2 = _mm256_loadu_si256((const __m256i *)(src + (-2*src_pitch)));
        ySrc1YM1 = _mm256_loadu_si256((const __m256i *)(src + (-1*src_pitch)));
        ySrc1YP1 = _mm256_loadu_si256((const __m256i *)(src + ( 1*src_pitch)));
        ySrc1YP2 = _mm256_loadu_si256((const __m256i *)(src + ( 2*src_pitch)));
        ySrc2    = _mm256_loadu_si256((const __m256i *)(src +             32));

        __m256i xY = edgelevel_avx2_8(ySrc1YM2, ySrc1YM1, ySrc0, ySrc1, ySrc2, ySrc1YP1, ySrc1YP2, yThreshold, yStrength, yWc, yBc);
        ySrc0 = ySrc1;
        ySrc1 = ySrc2;

        _mm256_storeu_si256((__m256i *)(dst + 0), xY);
    }
    const int loop_remain = w & (loopsize - 1);
    if (loop_remain) {
        __m256i xY = edgelevel_avx2_8(ySrc1YM2, ySrc1YM1, ySrc0, ySrc1, ySrc2, ySrc1YP1, ySrc1YP2, yThreshold, yStrength, yWc, yBc);

        alignas(32) uint8_t buf[32];
        _mm256_storeu_si256((__m256i *)buf, xY);
        uint8_t *ptr_dst = (uint8_t *)dst;
        for (int i = 0; i < loop_remain; i++) {
            ptr_dst[i] = buf[i];
        }
    }
    *(short *)dst_line = *(short *)src_line;
    *(short *)(dst_line + w - 2) = *(short *)(src_line + w - 2);
}

void edgelevel_func_mt_avx2_8_avisynth(thread_t *th) {
    const int dst_pitch = th->buf->dst_pitch[0];
    const int src_pitch = th->buf->src_pitch[0];
    const int h = th->buf->src_height[0], w = th->buf->src_width[0];
    const int strength = th->prm.strength; //これはスケーリングが必要ない
    const int threshold = th->prm.thrs >> 1;
    const int bc = th->prm.bc;
    const int wc = th->prm.wc;

    const __m256i yStrength = _mm256_set1_epi16((short)strength);
    const __m256i yThreshold = _mm256_set1_epi8((char)threshold);
    const __m256i yBc = _mm256_set1_epi8((char)bc);
    const __m256i yWc = _mm256_set1_epi8((char)wc);

    const int y_start = (h * th->thread_id    ) / th->threads;
    const int y_end   = (h * (th->thread_id+1)) / th->threads;

    for (int y = y_start; y < y_end; y++) {
        const char *src = (const char *)th->buf->src_ptr[0] + src_pitch * y;
        char *dst = (char *)th->buf->dst_ptr[0] + dst_pitch * y;
        if (y < 2 || h - 3 < y) {
            memcpy(dst, src, w);
        } else {
            edgelevel_avx2_8_line(dst, dst_pitch, src, src_pitch, w, yThreshold, yStrength, yWc, yBc);
        }
    }
    _mm256_zeroupper();
}

static __forceinline __m256i edgelevel_avx2_16(
    const __m256i &ySrc1YM2, const __m256i &ySrc1YM1,
    const __m256i &ySrc0, const __m256i &ySrc1, const __m256i &ySrc2,
    const __m256i &ySrc1YP1, const __m256i &ySrc1YP2,
    const __m256i &yThreshold, const __m256i &yStrength, const __m256i &yWc, const __m256i &yBc) {
    //周辺近傍の最大と最小を縦方向・横方向に求める
    __m256i yVmax, yVmin;
    yVmax = _mm256_max_epu16(ySrc1, ySrc1YM2);
    yVmin = _mm256_min_epu16(ySrc1, ySrc1YM2);
    yVmax = _mm256_max_epu16(yVmax, ySrc1YM1);
    yVmin = _mm256_min_epu16(yVmin, ySrc1YM1);
    yVmax = _mm256_max_epu16(yVmax, ySrc1YP1);
    yVmin = _mm256_min_epu16(yVmin, ySrc1YP1);
    yVmax = _mm256_max_epu16(yVmax, ySrc1YP2);
    yVmin = _mm256_min_epu16(yVmin, ySrc1YP2);

    __m256i ySrc1XM2 = _mm256_alignr256_epi8(ySrc1, ySrc0, 32-4);
    __m256i ySrc1XM1 = _mm256_alignr256_epi8(ySrc1, ySrc0, 32-2);
    __m256i ySrc1XP1 = _mm256_alignr256_epi8(ySrc2, ySrc1, 2);
    __m256i ySrc1XP2 = _mm256_alignr256_epi8(ySrc2, ySrc1, 4);
    __m256i yMax, yMin;
    yMax  = _mm256_max_epu16(ySrc1, ySrc1XM2);
    yMin  = _mm256_min_epu16(ySrc1, ySrc1XM2);
    yMax  = _mm256_max_epu16(yMax, ySrc1XM1);
    yMin  = _mm256_min_epu16(yMin, ySrc1XM1);
    yMax  = _mm256_max_epu16(yMax, ySrc1XP1);
    yMin  = _mm256_min_epu16(yMin, ySrc1XP1);
    yMax  = _mm256_max_epu16(yMax, ySrc1XP2);
    yMin  = _mm256_min_epu16(yMin, ySrc1XP2);

    //if (max - min < vmax - vmin) { max = vmax, min = vmin; }
    __m256i yMask = _mm256_cmpgt_epu16(_mm256_sub_epi16(yVmax, yVmin), _mm256_sub_epi16(yMax, yMin));
    yMax  = _mm256_blendv_epi8(yMax, yVmax, yMask);
    yMin  = _mm256_blendv_epi8(yMin, yVmin, yMask);

    //avg = (min + max) >> 1;
    __m256i yAvg = _mm256_avg_epu16(yMax, yMin);

    //if (max - min > thrs)
    yMask = _mm256_cmpgt_epu16(_mm256_subs_epi16(yMax, yMin), yThreshold);

    //if (src->y == max) max += wc * 2;
    //else max += wc;
    __m256i yMaskMax = _mm256_cmpeq_epi16(ySrc1, yMax);
    yMax  = _mm256_adds_epu16(yMax, yWc);
    yMax  = _mm256_adds_epu16(yMax, _mm256_and_si256(yWc, yMaskMax));

    //if (src->y == min) min -= bc * 2;
    //else  min -= bc;
    __m256i yMaskMin = _mm256_cmpeq_epi16(ySrc1, yMin);
    yMin  = _mm256_subs_epu16(yMin, yBc);
    yMin  = _mm256_subs_epu16(yMin, _mm256_and_si256(yBc, yMaskMin));

    //dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> (4)) ), min ), max );
    __m256i y0, y1;
    //必ず -32768～32767の範囲に収める
    //Avgはminとmaxの平均なので、かならずこの範囲に収まる
    y1    = _mm256_subs_epi16(yAvg, ySrc1);
    y0    = _mm256_unpacklo_epi16(y1, y1);
    y1    = _mm256_unpackhi_epi16(y1, y1);
    y0    = _mm256_madd_epi16(y0, yStrength);
    y1    = _mm256_madd_epi16(y1, yStrength);
    y0    = _mm256_srai_epi32(y0, 4);
    y1    = _mm256_srai_epi32(y1, 4);
    y0    = _mm256_add_epi32(y0, _mm256_unpacklo_epi16(ySrc1, _mm256_setzero_si256()));
    y1    = _mm256_add_epi32(y1, _mm256_unpackhi_epi16(ySrc1, _mm256_setzero_si256()));
    y0    = _mm256_packus_epi32(y0, y1);
    y0    = _mm256_max_epu16(y0, yMin);
    y0    = _mm256_min_epu16(y0, yMax);

    return _mm256_blendv_epi8(ySrc1, y0, yMask);
}

static __forceinline void edgelevel_avx2_16_line(
    char *dst_line, const int dst_pitch,
    const char *src_line, const int src_pitch, int w,
    const __m256i& yThreshold, const __m256i& yStrength, const __m256i& yWc, const __m256i& yBc) {
    char *dst = dst_line;
    const char *src = src_line;
    __m256i ySrc0 = get_previous_2_y_pixels_256(src);
    __m256i ySrc1 = _mm256_loadu_si256((const __m256i *)(src));
    __m256i ySrc1YM2, ySrc1YM1, ySrc1YP1, ySrc1YP2, ySrc2;
    const int loopsize = 16;
    const char *src_fin = src + (w & (~(loopsize-1))) * sizeof(uint16_t);
    for ( ; src < src_fin; src += 32, dst += 32) {
        //周辺近傍の最大と最小を縦方向・横方向に求める
        ySrc1YM2 = _mm256_loadu_si256((const __m256i *)(src + (-2*src_pitch)));
        ySrc1YM1 = _mm256_loadu_si256((const __m256i *)(src + (-1*src_pitch)));
        ySrc1YP1 = _mm256_loadu_si256((const __m256i *)(src + ( 1*src_pitch)));
        ySrc1YP2 = _mm256_loadu_si256((const __m256i *)(src + ( 2*src_pitch)));
        ySrc2    = _mm256_loadu_si256((const __m256i *)(src +             32));

        __m256i yY = edgelevel_avx2_16(ySrc1YM2, ySrc1YM1, ySrc0, ySrc1, ySrc2, ySrc1YP1, ySrc1YP2, yThreshold, yStrength, yWc, yBc);
        ySrc0 = ySrc1;
        ySrc1 = ySrc2;

        _mm256_storeu_si256((__m256i *)dst, yY);
    }
    const int loop_remain = w & (loopsize - 1);
    if (loop_remain) {
        __m256i yY = edgelevel_avx2_16(ySrc1YM2, ySrc1YM1, ySrc0, ySrc1, ySrc2, ySrc1YP1, ySrc1YP2, yThreshold, yStrength, yWc, yBc);

        alignas(32) uint16_t buf[16];
        _mm256_storeu_si256((__m256i *)buf, yY);
        uint16_t *ptr_dst = (uint16_t *)dst;
        for (int i = 0; i < loop_remain; i++) {
            ptr_dst[i] = buf[i];
        }
    }
    *(int *)dst_line = *(int *)src_line;
    *(int *)(dst_line + w - 4) = *(int *)(src_line + w - 4);
}

void edgelevel_func_mt_avx2_16_avisynth(thread_t *th) {
    const int dst_pitch = th->buf->dst_pitch[0];
    const int src_pitch = th->buf->src_pitch[0];
    const int w = th->buf->src_width[0];
    const int h = th->buf->src_height[0];
    const int bit_depth = th->prm.bit_depth;

    const int str = th->prm.strength; //これはスケーリングが必要ない
    const int thrs = (th->prm.thrs << 7) >> (16 - bit_depth);
    const int bc = th->prm.bc << (bit_depth - 8);
    const int wc = th->prm.wc << (bit_depth - 8);
    const __m256i yStrength = _mm256_unpacklo_epi16(_mm256_setzero_si256(), _mm256_set1_epi16((short)(-1 * str)));
    const __m256i yThreshold = _mm256_set1_epi16((short)thrs);
    const __m256i yBc = _mm256_set1_epi16((short)bc);
    const __m256i yWc = _mm256_set1_epi16((short)wc);

    const int y_start = (h * th->thread_id    ) / th->threads;
    const int y_end   = (h * (th->thread_id+1)) / th->threads;

    const char *line_src = (const char *)th->buf->src_ptr[0] + y_start * src_pitch;
    char *line_dst       = (char *)th->buf->dst_ptr[0]       + y_start * dst_pitch;
    for (int y = y_start; y < y_end; y++, line_src += src_pitch, line_dst += dst_pitch) {
        const char *src = line_src;
        char *dst = line_dst;
        if (y < 2 || h - 3 < y) {
            memcpy(dst, src, w * sizeof(uint16_t));
        } else {
            edgelevel_avx2_16_line(dst, dst_pitch, src, src_pitch, w, yThreshold, yStrength, yWc, yBc);
        }
    }
    _mm256_zeroupper();
}

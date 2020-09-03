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
#include <emmintrin.h>
#include <process.h>
#pragma comment(lib, "winmm.lib")
#include "edgelevel.h"

static inline __m128i _mm_cmpgt_epu8(__m128i x, __m128i y) {
    // Returns 0xFF where x > y:
    return _mm_andnot_si128(
        _mm_cmpeq_epi8(x, y),
        _mm_cmpeq_epi8(_mm_max_epu8(x, y), x)
    );
}
static inline __m128i _mm_cmple_epu16(__m128i x, __m128i y) {
    // Returns 0xFFFF where x <= y:
    return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}
static inline __m128i _mm_cmpgt_epu16(__m128i x, __m128i y) {
    // Returns 0xFFFF where x > y:
    return _mm_andnot_si128(_mm_cmpeq_epi16(x, y), _mm_cmple_epu16(y, x));
}
static inline __m128i _mm_blendv_si128(__m128i x, __m128i y, __m128i mask) {
    // Replace bit in x with bit in y when matching bit in mask is set:
    return _mm_or_si128(_mm_andnot_si128(mask, x), _mm_and_si128(mask, y));
}
static inline __m128i _mm_max_epu16(__m128i x, __m128i y) {
    // Returns x where x >= y, else y:
    return _mm_blendv_si128(x, y, _mm_cmple_epu16(x, y));
}
static inline __m128i _mm_min_epu16(__m128i x, __m128i y) {
    // Returns x where x <= y, else y:
    return _mm_blendv_si128(y, x, _mm_cmple_epu16(x, y));
}

static inline __m128i _mm_packus_epi32(__m128i a, __m128i b) {
    const static __m128i val_32 = _mm_set1_epi32(0x8000);
    const static __m128i val_16 = _mm_set1_epi16(-32768);
    a = _mm_sub_epi32(a, val_32);
    b = _mm_sub_epi32(b, val_32);
    a = _mm_packs_epi32(a, b);
    a = _mm_add_epi16(a, val_16);
    return a;
}

static __forceinline __m128i get_previous_2_y_pixels_128(const char *src) {
    return _mm_loadu_si128((const __m128i *)(src - 16));
}

#define pblendvb_SSE2(a,b,mask) _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) )
#define palignr_SSE2(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-(i)), _mm_srli_si128(b, (i)) )

static __forceinline __m128i edgelevel_sse2_8(
    const __m128i &xSrc1YM2, const __m128i &xSrc1YM1,
    const __m128i &xSrc0, const __m128i &xSrc1, const __m128i &xSrc2,
    const __m128i &xSrc1YP1, const __m128i &xSrc1YP2,
    const __m128i &xThreshold, const __m128i &xStrength, const __m128i &xWc, const __m128i &xBc) {

    __m128i xVmax, xVmin;
    xVmax = _mm_max_epu8(xSrc1, xSrc1YM2);
    xVmin = _mm_min_epu8(xSrc1, xSrc1YM2);
    xVmax = _mm_max_epu8(xVmax, xSrc1YM1);
    xVmin = _mm_min_epu8(xVmin, xSrc1YM1);
    xVmax = _mm_max_epu8(xVmax, xSrc1YP1);
    xVmin = _mm_min_epu8(xVmin, xSrc1YP1);
    xVmax = _mm_max_epu8(xVmax, xSrc1YP2);
    xVmin = _mm_min_epu8(xVmin, xSrc1YP2);

    __m128i xSrc1XM2 = palignr_SSE2(xSrc1, xSrc0, 16 - 2);
    __m128i xSrc1XM1 = palignr_SSE2(xSrc1, xSrc0, 16 - 1);
    __m128i xSrc1XP1 = palignr_SSE2(xSrc2, xSrc1, 1);
    __m128i xSrc1XP2 = palignr_SSE2(xSrc2, xSrc1, 2);
    __m128i xMax, xMin;
    xMax = _mm_max_epu8(xSrc1, xSrc1XM2);
    xMin = _mm_min_epu8(xSrc1, xSrc1XM2);
    xMax = _mm_max_epu8(xMax, xSrc1XM1);
    xMin = _mm_min_epu8(xMin, xSrc1XM1);
    xMax = _mm_max_epu8(xMax, xSrc1XP1);
    xMin = _mm_min_epu8(xMin, xSrc1XP1);
    xMax = _mm_max_epu8(xMax, xSrc1XP2);
    xMin = _mm_min_epu8(xMin, xSrc1XP2);

    //if (max - min < vmax - vmin) { max = vmax; min = vmin; }
    __m128i xMask = _mm_cmpgt_epu8(_mm_sub_epi8(xVmax, xVmin), _mm_sub_epi8(xMax, xMin));
    xMax  = pblendvb_SSE2(xMax, xVmax, xMask);
    xMin  = pblendvb_SSE2(xMin, xVmin, xMask);

    //avg = (min + max + 1) >> 1;
    __m128i xAvg  = _mm_avg_epu8(xMax, xMin);

    //if (max - min > thrs)
    xMask = _mm_cmpgt_epu8(_mm_subs_epi8(xMax, xMin), xThreshold);

    //if (src->y == max) max += wc * 2;
    //else max += wc;
    __m128i xMaskMax = _mm_cmpeq_epi8(xSrc1, xMax);
    xMax  = _mm_adds_epu8(xMax, xWc);
    xMax  = _mm_adds_epu8(xMax, _mm_and_si128(xWc, xMaskMax));

    //if (src->y == min) min -= bc * 2;
    //else  min -= bc;
    __m128i xMaskMin = _mm_cmpeq_epi8(xSrc1, xMin);
    xMin  = _mm_subs_epu8(xMin, xBc);
    xMin  = _mm_subs_epu8(xMin, _mm_and_si128(xBc, xMaskMin));

    //dst[x] = (uint8_t)clamp(src[x] + (((src[x] - avg) * strength) >> 4), (std::max)(min, 0), (std::min)(max, 255));
    __m128i x0, x1;
    x0    = _mm_sub_epi16(_mm_unpacklo_epi8(xSrc1, _mm_setzero_si128()), _mm_unpacklo_epi8(xAvg, _mm_setzero_si128()));
    x1    = _mm_sub_epi16(_mm_unpackhi_epi8(xSrc1, _mm_setzero_si128()), _mm_unpackhi_epi8(xAvg, _mm_setzero_si128()));
    x0    = _mm_mullo_epi16(x0, xStrength);
    x1    = _mm_mullo_epi16(x1, xStrength);
    x0    = _mm_srai_epi16(x0, 4);
    x1    = _mm_srai_epi16(x1, 4);
    x0    = _mm_add_epi16(x0, _mm_unpacklo_epi8(xSrc1, _mm_setzero_si128()));
    x1    = _mm_add_epi16(x1, _mm_unpackhi_epi8(xSrc1, _mm_setzero_si128()));
    x0    = _mm_packus_epi16(x0, x1);
    x0    = _mm_min_epu8(x0, xMax);
    x0    = _mm_max_epu8(x0, xMin);

    return pblendvb_SSE2(xSrc1, x0, xMask);
}

static __forceinline void edgelevel_sse2_8_line(
    char *dst_line, const int dst_pitch,
    const char *src_line, const int src_pitch, int w,
    const __m128i &xThreshold, const __m128i &xStrength, const __m128i &xWc, const __m128i &xBc) {
    const char *src = src_line;
    char *dst = dst_line;
    __m128i xSrc0 = get_previous_2_y_pixels_128(src);
    __m128i xSrc1 = _mm_loadu_si128((const __m128i *)(src));
    __m128i xSrc1YM2, xSrc1YM1, xSrc1YP1, xSrc1YP2, xSrc2;
    const int loopsize = 16;
    const char *src_fin = src + (w & (~(loopsize - 1))) * sizeof(uint16_t);
    for (; src < src_fin; src += 16, dst += 16) {
        //周辺近傍の最大と最小を縦方向・横方向に求める
        xSrc1YM2 = _mm_loadu_si128((const __m128i *)(src + (-2*src_pitch)));
        xSrc1YM1 = _mm_loadu_si128((const __m128i *)(src + (-1*src_pitch)));
        xSrc1YP1 = _mm_loadu_si128((const __m128i *)(src + ( 1*src_pitch)));
        xSrc1YP2 = _mm_loadu_si128((const __m128i *)(src + ( 2*src_pitch)));
        xSrc2    = _mm_loadu_si128((const __m128i *)(src +             16));

        __m128i xY = edgelevel_sse2_8(xSrc1YM2, xSrc1YM1, xSrc0, xSrc1, xSrc2, xSrc1YP1, xSrc1YP2, xThreshold, xStrength, xWc, xBc);
        xSrc0 = xSrc1;
        xSrc1 = xSrc2;

        _mm_storeu_si128((__m128i *)(dst + 0), xY);
    }
    const int loop_remain = w & (loopsize - 1);
    if (loop_remain) {
        __m128i xY = edgelevel_sse2_8(xSrc1YM2, xSrc1YM1, xSrc0, xSrc1, xSrc2, xSrc1YP1, xSrc1YP2, xThreshold, xStrength, xWc, xBc);

        alignas(32) uint8_t buf[16];
        _mm_storeu_si128((__m128i *)buf, xY);
        uint8_t *ptr_dst = (uint8_t *)dst;
        for (int i = 0; i < loop_remain; i++) {
            ptr_dst[i] = buf[i];
        }
    }
    *(short *)dst_line = *(short *)src_line;
    *(short *)(dst_line + w - 2) = *(short *)(src_line + w - 2);
}

static void edgelevel_func_mt_sse2_8_avisynth(thread_t *th) {
    const int dst_pitch = th->buf->dst_pitch[0];
    const int src_pitch = th->buf->src_pitch[0];
    const int h = th->buf->src_height[0], w = th->buf->src_width[0];
    const int strength = th->prm.strength; //これはスケーリングが必要ない
    const int threshold = th->prm.thrs >> 1;
    const int bc = th->prm.bc;
    const int wc = th->prm.wc;

    const __m128i xStrength = _mm_set1_epi16((short)strength);
    const __m128i xThreshold = _mm_set1_epi8((char)threshold);
    const __m128i xBc = _mm_set1_epi8((char)bc);
    const __m128i xWc = _mm_set1_epi8((char)wc);

    const int y_start = (h * th->thread_id    ) / th->threads;
    const int y_end   = (h * (th->thread_id+1)) / th->threads;

    for (int y = y_start; y < y_end; y++) {
        const char *src = (const char *)th->buf->src_ptr[0] + src_pitch * y;
        char *dst = (char *)th->buf->dst_ptr[0] + dst_pitch * y;
        if (y < 2 || h - 3 < y) {
            memcpy(dst, src, w);
        } else {
            edgelevel_sse2_8_line(dst, dst_pitch, src, src_pitch, w, xThreshold, xStrength, xWc, xBc);
        }
    }
}

static __forceinline __m128i edgelevel_sse2_16(
    const __m128i &xSrc1YM2, const __m128i &xSrc1YM1,
    const __m128i &xSrc0, const __m128i &xSrc1, const __m128i &xSrc2,
    const __m128i &xSrc1YP1, const __m128i &xSrc1YP2,
    const __m128i &xThreshold, const __m128i &xStrength, const __m128i &xWc, const __m128i &xBc) {
    //周辺近傍の最大と最小を縦方向・横方向に求める
    __m128i yVmax, yVmin;
    yVmax = _mm_max_epu16(xSrc1, xSrc1YM2);
    yVmin = _mm_min_epu16(xSrc1, xSrc1YM2);
    yVmax = _mm_max_epu16(yVmax, xSrc1YM1);
    yVmin = _mm_min_epu16(yVmin, xSrc1YM1);
    yVmax = _mm_max_epu16(yVmax, xSrc1YP1);
    yVmin = _mm_min_epu16(yVmin, xSrc1YP1);
    yVmax = _mm_max_epu16(yVmax, xSrc1YP2);
    yVmin = _mm_min_epu16(yVmin, xSrc1YP2);

    __m128i xSrc1XM2 = palignr_SSE2(xSrc1, xSrc0, 16-4);
    __m128i xSrc1XM1 = palignr_SSE2(xSrc1, xSrc0, 16-2);
    __m128i xSrc1XP1 = palignr_SSE2(xSrc2, xSrc1, 2);
    __m128i xSrc1XP2 = palignr_SSE2(xSrc2, xSrc1, 4);
    __m128i xMax, xMin;
    xMax  = _mm_max_epu16(xSrc1, xSrc1XM2);
    xMin  = _mm_min_epu16(xSrc1, xSrc1XM2);
    xMax  = _mm_max_epu16(xMax, xSrc1XM1);
    xMin  = _mm_min_epu16(xMin, xSrc1XM1);
    xMax  = _mm_max_epu16(xMax, xSrc1XP1);
    xMin  = _mm_min_epu16(xMin, xSrc1XP1);
    xMax  = _mm_max_epu16(xMax, xSrc1XP2);
    xMin  = _mm_min_epu16(xMin, xSrc1XP2);

    //if (max - min < vmax - vmin) { max = vmax, min = vmin; }
    __m128i yMask = _mm_cmpgt_epu16(_mm_sub_epi16(yVmax, yVmin), _mm_sub_epi16(xMax, xMin));
    xMax  = pblendvb_SSE2(xMax, yVmax, yMask);
    xMin  = pblendvb_SSE2(xMin, yVmin, yMask);

    //avg = (min + max) >> 1;
    __m128i yAvg = _mm_avg_epu16(xMax, xMin);

    //if (max - min > thrs)
    yMask = _mm_cmpgt_epu16(_mm_subs_epi16(xMax, xMin), xThreshold);

    //if (src->y == max) max += wc * 2;
    //else max += wc;
    __m128i yMaskMax = _mm_cmpeq_epi16(xSrc1, xMax);
    xMax  = _mm_adds_epu16(xMax, xWc);
    xMax  = _mm_adds_epu16(xMax, _mm_and_si128(xWc, yMaskMax));

    //if (src->y == min) min -= bc * 2;
    //else  min -= bc;
    __m128i yMaskMin = _mm_cmpeq_epi16(xSrc1, xMin);
    xMin  = _mm_subs_epu16(xMin, xBc);
    xMin  = _mm_subs_epu16(xMin, _mm_and_si128(xBc, yMaskMin));

    //dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> (4)) ), min ), max );
    __m128i y0, y1;
    //必ず -32768～32767の範囲に収める
    //Avgはminとmaxの平均なので、かならずこの範囲に収まる
    y1    = _mm_subs_epi16(yAvg, xSrc1);
    y0    = _mm_unpacklo_epi16(y1, y1);
    y1    = _mm_unpackhi_epi16(y1, y1);
    y0    = _mm_madd_epi16(y0, xStrength);
    y1    = _mm_madd_epi16(y1, xStrength);
    y0    = _mm_srai_epi32(y0, 4);
    y1    = _mm_srai_epi32(y1, 4);
    y0    = _mm_add_epi32(y0, _mm_unpacklo_epi16(xSrc1, _mm_setzero_si128()));
    y1    = _mm_add_epi32(y1, _mm_unpackhi_epi16(xSrc1, _mm_setzero_si128()));
    y0    = _mm_packus_epi32(y0, y1);
    y0    = _mm_max_epu16(y0, xMin);
    y0    = _mm_min_epu16(y0, xMax);

    return pblendvb_SSE2(xSrc1, y0, yMask);
}

static __forceinline void edgelevel_sse2_16_line(
    char *dst_line, const int dst_pitch,
    const char *src_line, const int src_pitch, int w,
    const __m128i& xThreshold, const __m128i& xStrength, const __m128i& xWc, const __m128i& xBc) {
    char *dst = dst_line;
    const char *src = src_line;
    __m128i xSrc0 = get_previous_2_y_pixels_128(src);
    __m128i xSrc1 = _mm_loadu_si128((const __m128i *)(src));
    __m128i xSrc1YM2, xSrc1YM1, xSrc1YP1, xSrc1YP2, xSrc2;
    const int loopsize = 8;
    const char *src_fin = src + (w & (~(loopsize-1))) * sizeof(uint16_t);
    for ( ; src < src_fin; src += 16, dst += 16) {
        //周辺近傍の最大と最小を縦方向・横方向に求める
        xSrc1YM2 = _mm_loadu_si128((const __m128i *)(src + (-2*src_pitch)));
        xSrc1YM1 = _mm_loadu_si128((const __m128i *)(src + (-1*src_pitch)));
        xSrc1YP1 = _mm_loadu_si128((const __m128i *)(src + ( 1*src_pitch)));
        xSrc1YP2 = _mm_loadu_si128((const __m128i *)(src + ( 2*src_pitch)));
        xSrc2    = _mm_loadu_si128((const __m128i *)(src +            16));

        __m128i yY = edgelevel_sse2_16(xSrc1YM2, xSrc1YM1, xSrc0, xSrc1, xSrc2, xSrc1YP1, xSrc1YP2, xThreshold, xStrength, xWc, xBc);
        xSrc0 = xSrc1;
        xSrc1 = xSrc2;

        _mm_storeu_si128((__m128i *)dst, yY);
    }
    const int loop_remain = w & (loopsize - 1);
    if (loop_remain) {
        __m128i yY = edgelevel_sse2_16(xSrc1YM2, xSrc1YM1, xSrc0, xSrc1, xSrc2, xSrc1YP1, xSrc1YP2, xThreshold, xStrength, xWc, xBc);

        alignas(16) uint16_t buf[8];
        _mm_storeu_si128((__m128i *)buf, yY);
        uint16_t *ptr_dst = (uint16_t *)dst;
        for (int i = 0; i < loop_remain; i++) {
            ptr_dst[i] = buf[i];
        }
    }
    *(int *)dst_line = *(int *)src_line;
    *(int *)(dst_line + w - 4) = *(int *)(src_line + w - 4);
}

static void edgelevel_func_mt_sse2_16_avisynth(thread_t *th) {
    const int dst_pitch = th->buf->dst_pitch[0];
    const int src_pitch = th->buf->src_pitch[0];
    const int w = th->buf->src_width[0];
    const int h = th->buf->src_height[0];
    const int bit_depth = th->prm.bit_depth;

    const int str = th->prm.strength; //これはスケーリングが必要ない
    const int thrs = (th->prm.thrs << 7) >> (16 - bit_depth);
    const int bc = th->prm.bc << (bit_depth - 8);
    const int wc = th->prm.wc << (bit_depth - 8);
    const __m128i xStrength = _mm_unpacklo_epi16(_mm_setzero_si128(), _mm_set1_epi16((short)(-1 * str)));
    const __m128i xThreshold = _mm_set1_epi16((short)thrs);
    const __m128i xBc = _mm_set1_epi16((short)bc);
    const __m128i xWc = _mm_set1_epi16((short)wc);

    const int y_start = (h * th->thread_id) / th->threads;
    const int y_end = (h * (th->thread_id + 1)) / th->threads;

    const char *line_src = (const char *)th->buf->src_ptr[0] + y_start * src_pitch;
    char *line_dst = (char *)th->buf->dst_ptr[0] + y_start * dst_pitch;
    for (int y = y_start; y < y_end; y++, line_src += src_pitch, line_dst += dst_pitch) {
        const char *src = line_src;
        char *dst = line_dst;
        if (y < 2 || h - 3 < y) {
            memcpy(dst, src, w * sizeof(uint16_t));
        } else {
            edgelevel_sse2_16_line(dst, dst_pitch, src, src_pitch, w, xThreshold, xStrength, xWc, xBc);
        }
    }
}

void edgelevel_func_mt_avx2_8_avisynth(thread_t *th);
void edgelevel_func_mt_avx2_16_avisynth(thread_t *th);

unsigned int __stdcall RunThread(void *arg) {
    thread_t *th = (thread_t *)arg;
    WaitForSingleObject(th->he_start, INFINITE);
    while (!th->abort) {
        if (th->prm.bit_depth > 8) {
            if (th->prm.avx2) {
                edgelevel_func_mt_avx2_16_avisynth(th);
            } else {
                edgelevel_func_mt_sse2_16_avisynth(th);
            }
        } else {
            if (th->prm.avx2) {
                edgelevel_func_mt_avx2_8_avisynth(th);
            } else {
                edgelevel_func_mt_sse2_8_avisynth(th);
            }
        }
        SetEvent(th->he_fin);
        WaitForSingleObject(th->he_start, INFINITE);
    }
    _endthreadex(0);
    return 0;
}

bool CreateThreads(int threads, frame_buf_t *_buf, edegelevel_prm_t *_prm, mt_control_t *mt_control) {
    if (!check_range(threads, 0, MAX_THREADS))
        return FALSE;

    bool ret = TRUE;
    mt_control->threads = threads;
    for (int i = 0; i < threads; i++) {
        mt_control->th_prm[i].abort = FALSE;
        mt_control->th_prm[i].thread_id = i;
        mt_control->th_prm[i].threads = threads;
        mt_control->th_prm[i].buf = _buf;
        mt_control->th_prm[i].prm = *_prm;
        if (NULL == (mt_control->th_prm[i].he_start = CreateEvent(NULL, FALSE, FALSE, NULL)) ||
            NULL == (mt_control->th_prm[i].he_fin   = CreateEvent(NULL, FALSE, FALSE, NULL))) {
            ret = FALSE; break;
        }
        mt_control->he_fin_copy[i] = mt_control->th_prm[i].he_fin;
        if (NULL == (mt_control->th_threads[i] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &mt_control->th_prm[i], NULL, NULL))) {
            ret = FALSE; break;
        }
    }
    return ret;
}

void FinishThreads(mt_control_t *mt_control) {
    if (!mt_control)
        return;

    for (int i = 0; i < mt_control->threads; i++) {
        mt_control->th_prm[i].abort = TRUE;
        SetEvent(mt_control->th_prm[i].he_start);
    }
    WaitForMultipleObjects(mt_control->threads, mt_control->th_threads, TRUE, INFINITE);
    for (int i = 0; i < mt_control->threads; i++) {
        CloseHandle(mt_control->th_prm[i].he_start);
        CloseHandle(mt_control->th_prm[i].he_fin);
        CloseHandle(mt_control->th_threads[i]);
    }
    return;
}

void StartProcessFrame(mt_control_t *mt_control) {
    for (int i = 0; i < mt_control->threads; i++) {
        SetEvent(mt_control->th_prm[i].he_start);
    }
    WaitForMultipleObjects(mt_control->threads, mt_control->he_fin_copy, TRUE, INFINITE);
}


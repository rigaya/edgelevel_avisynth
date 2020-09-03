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
#include <stdio.h>
#include <algorithm>
#include <emmintrin.h>

#pragma warning(push)
#pragma warning(disable:4100)
#pragma warning(disable:4244)
#pragma warning(disable:4512)
#include "avisynth.h"
#pragma warning(pop)

#include "edgelevel.h"

const AVS_Linkage *AVS_linkage = 0;

static void copy_y_from_yuy2_sse2(BYTE *dst_ptr, int dst_pitch, const BYTE *src_ptr, int src_pitch, int width, int height) {
    const BYTE *src, *src_fin;
    BYTE *dst;
    __m128i x0, x1, x2;
    for (int y = 0; y < height; y++) {
        dst = dst_ptr + dst_pitch * y;
        src = src_ptr + src_pitch * y;
        src_fin = src + width * 2;
        for (;src < src_fin; src += 32, dst += 16) {
            x0 = _mm_loadu_si128((__m128i *)(src+ 0));
            x1 = _mm_loadu_si128((__m128i *)(src+16));
            x2 = _mm_unpacklo_epi8(x0, x1);
            x1 = _mm_unpackhi_epi8(x0, x1);
            x0 = _mm_unpacklo_epi8(x2, x1);
            x1 = _mm_unpackhi_epi8(x2, x1);
            x2 = _mm_unpacklo_epi8(x0, x1);
            x1 = _mm_unpackhi_epi8(x0, x1);
            _mm_storeu_si128((__m128i *)dst, _mm_unpacklo_epi8(x2, x1));
        }
    }
}

static void copy_to_yuy2_sse2(BYTE *dst_ptr, int dst_pitch, const BYTE *y_src_ptr, int y_src_pitch, const BYTE *uv_src_ptr, int uv_src_pitch, int width, int height) {
    const BYTE *y_src, *y_src_fin, *uv_src;
    BYTE *dst;
    __m128i x0, x1, x2;
    __m128i x3 = _mm_setzero_si128();
    for (int y = 0; y < height; y++) {
        dst = dst_ptr + dst_pitch * y;
        y_src = y_src_ptr + y_src_pitch * y;
        uv_src = uv_src_ptr + uv_src_pitch * y;
        y_src_fin = y_src + width;
        for (;y_src < y_src_fin; y_src += 16, uv_src += 32, dst += 32) {
            x0 = _mm_loadu_si128((__m128i *)(y_src+ 0)); //Y
            x1 = _mm_loadu_si128((__m128i *)(uv_src+ 0)); //UV
            x2 = _mm_loadu_si128((__m128i *)(uv_src+16)); //UV
            x1 = _mm_and_si128(x1, _mm_unpacklo_epi8(_mm_setzero_si128(), _mm_cmpeq_epi8(x3, x3)));
            x2 = _mm_and_si128(x2, _mm_unpacklo_epi8(_mm_setzero_si128(), _mm_cmpeq_epi8(x3, x3)));
            _mm_storeu_si128((__m128i *)(dst+ 0), _mm_or_si128(_mm_unpacklo_epi8(x0, _mm_setzero_si128()), x1));
            _mm_storeu_si128((__m128i *)(dst+16), _mm_or_si128(_mm_unpackhi_epi8(x0, _mm_setzero_si128()), x2));
        }
    }
}

#pragma warning(push)
#pragma warning(disable:4512)
class edgelevel : public GenericVideoFilter {
private:
    const int strength;  // 特性
    const int thrs;      // 閾値
    const int bc;        // 黒補正
    const int wc;        // 白補正
    edegelevel_prm_t prm;
    mt_control_t mt_control;
    int threads;         // スレッド数
    frame_buf_t buf;

public:
    edgelevel(PClip _child, int _strength, int _thrs, int _bc, int _wc, int _threads, IScriptEnvironment *env);
    ~edgelevel();
    BOOL AllocBuffer();
    void FreeBuffer();
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
};
#pragma warning(pop)

BOOL edgelevel::AllocBuffer() {
    if (vi.IsYUY2()) {
        buf.planes = 1;
        buf.src_pitch[0] = vi.width;
        buf.dst_pitch[0] = vi.width;
        int frame_buffer_size = ((vi.width * vi.height) + 15) & ~15;
        if (NULL == (buf.dst_ptr[0] = (BYTE *)_aligned_malloc(frame_buffer_size, 16)) ||
            NULL == (buf.src_ptr[0] = (BYTE *)_aligned_malloc(frame_buffer_size, 16))) {
                return FALSE;
        }
    }
    return TRUE;
}

void edgelevel::FreeBuffer() {
    if (vi.IsYUY2() && buf.planes) {
        if (buf.dst_ptr[0]) {_aligned_free(buf.dst_ptr[0]);          buf.dst_ptr[0] = NULL; }
        if (buf.src_ptr[0]) { _aligned_free((void *)buf.src_ptr[0]); buf.src_ptr[0] = NULL; }
    }
    buf.planes = 0;
}

edgelevel::edgelevel(PClip _child, int _strength, int _thrs, int _bc, int _wc, int _threads, IScriptEnvironment *env)
    : GenericVideoFilter(_child), strength(_strength), thrs(_thrs), bc(_bc), wc(_wc), threads(_threads) {
    if (!(env->GetCPUFlags() & CPUF_SSE2))     env->ThrowError("edgelevel: requires sse2 support.");
    if (!(vi.IsYUY2() || vi.IsYV12()))         env->ThrowError("edgelevel: input colorspace must be YUY2 or YV12.");
    if (!check_range(strength, -31, 31))       env->ThrowError("edgelevel: strength should be -31 - 31.");
    if (!check_range(thrs, 0, 255))            env->ThrowError("edgelevel: threshold should be 0 - 255.");
    if (!check_range(bc, 0, 31))               env->ThrowError("edgelevel: bc should be 0 - 31.");
    if (!check_range(wc, 0, 31))               env->ThrowError("edgelevel: wc should be 0 - 31.");
    if (!check_range(threads, 0, MAX_THREADS)) env->ThrowError("edgelevel: threads should be 0 - %d.", MAX_THREADS);

    prm.strength = strength;
    prm.thrs = thrs;
    prm.wc = wc;
    prm.bc = bc;

    ZeroMemory(&buf, sizeof(frame_buf_t));

    if (!threads) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        threads = (std::min)((int)si.dwNumberOfProcessors, MAX_THREADS);
    }
    if (!AllocBuffer()) env->ThrowError("edgelevel: failed to allocate memory.");
    if (!CreateThreads(threads, &buf, &prm, &mt_control)) env->ThrowError("edgelevel: failed to create threads.");
}

edgelevel::~edgelevel() {
    FinishThreads(&mt_control);
    FreeBuffer();
}

PVideoFrame __stdcall edgelevel::GetFrame(int n, IScriptEnvironment *env) {
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    if (vi.IsYUY2()) {
        copy_y_from_yuy2_sse2(buf.src_ptr[0], buf.src_pitch[0], src->GetReadPtr(), src->GetPitch(), src->GetRowSize() / 2, src->GetHeight());

        buf.src_width[0] = src->GetRowSize() / 2;
        buf.dst_width[0] = dst->GetRowSize() / 2;
        buf.src_height[0] = src->GetHeight();
        buf.dst_height[0] = dst->GetHeight();
    } else {
        //YV12
        buf.src_ptr[0] = (BYTE *)src->GetReadPtr();
        buf.dst_ptr[0] = dst->GetWritePtr();
        buf.src_pitch[0] = src->GetPitch();
        buf.dst_pitch[0] = dst->GetPitch();
        buf.src_width[0] = src->GetRowSize();
        buf.src_height[0] = src->GetHeight();
        buf.dst_width[0] = dst->GetRowSize();
        buf.dst_height[0] = dst->GetHeight();
    }
    //エッジレベル調整
    StartProcessFrame(&mt_control);

    if (vi.IsYUY2()) {
        copy_to_yuy2_sse2(dst->GetWritePtr(), dst->GetPitch(), buf.dst_ptr[0], buf.dst_pitch[0], src->GetReadPtr(), src->GetPitch(), src->GetRowSize() / 2, src->GetHeight());
    } else {
        buf.src_ptr[0] = buf.dst_ptr[0] = NULL;
        env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U), src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
        env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V), src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
    }
    return dst;
}

#pragma warning(push)
#pragma warning(disable:4100)
AVSValue __cdecl Create_edgelevel(AVSValue args, void *user_data, IScriptEnvironment *env) {
    return new edgelevel(args[0].AsClip(), args[1].AsInt(10), args[2].AsInt(16), args[3].AsInt(0), args[4].AsInt(0), args[5].AsInt(0), env);
}
#pragma warning(pop)

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("edgelevel", "c[strength]i[threshold]i[bc]i[wc]i[threads]i[asm]i", Create_edgelevel, 0);
    return "edgelevel for avisynth 0.00";
}

#include <Windows.h>
#include <algorithm>
#include <emmintrin.h>
#include <process.h>
#pragma comment(lib, "winmm.lib")
#include "edgelevel.h"

static void edgelevel_func_mt_sse2_avisynth(thread_t *th) {
#define pblendvb_SSE2(a,b,mask) _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) )
#define palignr_SSE2(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) )
	BYTE *src, *dst, *src_fin;
	const int dst_pitch = th->buf->dst_pitch[0];
	const int src_pitch = th->buf->src_pitch[0];
	const int h = th->buf->src_height[0], w = th->buf->src_width[0];
	const char strength = (char)th->prm.strength;
	const BYTE threshold = (BYTE)th->prm.thrs >> 1;
	const BYTE bc = (BYTE)th->prm.bc;
	const BYTE wc = (BYTE)th->prm.wc;
	__m128i x0, x1, x2, xVmax, xMax, xVmin, xMin, xAvg, xY, xMask;
	static const _declspec(align(16)) BYTE Array_MASK_FRAME_EDGE[]   = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

	const int y_start = (h * th->thread_id    ) / th->threads;
	const int y_end   = (h * (th->thread_id+1)) / th->threads;

	for (int y = y_start; y < y_end; y++) {
		src = th->buf->src_ptr[0] + src_pitch * y;
		dst = th->buf->dst_ptr[0] + dst_pitch * y;
		if (y < 2 || h - 3 < y) {
			memcpy(dst, src, w);
		} else {
			src_fin = src + w;
			BYTE *line_src = src;
			BYTE *line_dst = dst;
			for ( ; src < src_fin; src += 16, dst += 16) {
				//周辺近傍の最大と最小を縦方向・横方向に求める
				xVmax = _mm_loadu_si128((__m128i *)(src + (-2*src_pitch)));
				xVmin = xVmax;
				x0    = _mm_loadu_si128((__m128i *)(src + (-1*src_pitch)));
				xVmax = _mm_max_epu8(xVmax, x0);
				xVmin = _mm_min_epu8(xVmin, x0);
				x0    = _mm_loadu_si128((__m128i *)(src + ( 0 - 2)));
				x1    = _mm_loadu_si128((__m128i *)(src + (16 - 2)));
				xMax  = x0;
				xMin  = x0;
				x2    = palignr_SSE2(x1, x0, 1);
				xMax  = _mm_max_epu8(xMax, x2);
				xMin  = _mm_min_epu8(xMin, x2);
				xY    = palignr_SSE2(x1, x0, 2);
				xMax  = _mm_max_epu8(xMax, xY);
				xMin  = _mm_min_epu8(xMin, xY);
				xVmax = _mm_max_epu8(xVmax, xY);
				xVmin = _mm_min_epu8(xVmin, xY);
				x2    = palignr_SSE2(x1, x0, 3);
				xMax  = _mm_max_epu8(xMax, x2);
				xMin  = _mm_min_epu8(xMin, x2);
				x2    = palignr_SSE2(x1, x0, 4);
				xMax  = _mm_max_epu8(xMax, x2);
				xMin  = _mm_min_epu8(xMin, x2);
				x0    = _mm_load_si128((__m128i *)(src + (1*src_pitch)));
				xVmax = _mm_max_epu8(xVmax, x0);
				xVmin = _mm_min_epu8(xVmin, x0);
				x0    = _mm_load_si128((__m128i *)(src + (2*src_pitch)));
				xVmax = _mm_max_epu8(xVmax, x0);
				xVmin = _mm_min_epu8(xVmin, x0);

				//if (max - min < vmax - vmin) { max = vmax; min = vmin; }
				xMask = _mm_cmpgt_epi8(_mm_subs_epu8(xVmax, xVmin), _mm_subs_epu8(xMax, xMin));
				xMax  = pblendvb_SSE2(xMax, xVmax, xMask);
				xMin  = pblendvb_SSE2(xMin, xVmin, xMask);

				//avg = (min + max + 1) >> 1;
				xAvg  = _mm_avg_epu8(xMax, xMin);

				//if (max - min > thrs)
				xMask = _mm_cmpgt_epi8(_mm_subs_epi8(xMax, xMin), _mm_set1_epi8(threshold));

				//min -= bc * (1 + (src[x] == min))
				x1    = _mm_cmpeq_epi8(xY, xMax);
				xMax  = _mm_adds_epu8(xMax, _mm_and_si128(_mm_set1_epi8(wc), x1));
				xMax  = _mm_adds_epu8(xMax, _mm_set1_epi8(wc));

				//max += wc * (1 + (src[x] == max));
				x1    = _mm_cmpeq_epi8(xY, xMin);
				xMin  = _mm_subs_epu8(xMin, _mm_and_si128(_mm_set1_epi8(bc), x1));
				xMin  = _mm_subs_epu8(xMin, _mm_set1_epi8(bc));

				//dst[x] = (BYTE)clamp(src[x] + (((src[x] - avg) * strength) >> 4), (std::max)(min, 0), (std::min)(max, 255));
				x0    = _mm_sub_epi16(_mm_unpacklo_epi8(xY, _mm_setzero_si128()), _mm_unpacklo_epi8(xAvg, _mm_setzero_si128()));
				x1    = _mm_sub_epi16(_mm_unpackhi_epi8(xY, _mm_setzero_si128()), _mm_unpackhi_epi8(xAvg, _mm_setzero_si128()));
				x0    = _mm_mullo_epi16(x0, _mm_set1_epi16(strength));
				x1    = _mm_mullo_epi16(x1, _mm_set1_epi16(strength));
				x0    = _mm_srai_epi16(x0, 4);
				x1    = _mm_srai_epi16(x1, 4);
				x0    = _mm_add_epi16(x0, _mm_unpacklo_epi8(xY, _mm_setzero_si128()));
				x1    = _mm_add_epi16(x1, _mm_unpackhi_epi8(xY, _mm_setzero_si128()));
				x0    = _mm_packus_epi16(x0, x1);
				x0    = _mm_min_epu8(x0, xMax);
				x0    = _mm_max_epu8(x0, xMin);

				xY    = pblendvb_SSE2(xY, x0, xMask);

				_mm_storeu_si128((__m128i *)(dst +  0), xY);
			}
		    x0 = _mm_loadu_si128((__m128i *)line_src);
			x1 = _mm_loadu_si128((__m128i *)(line_src + w - 6));
			_mm_maskmoveu_si128(x0, _mm_load_si128((__m128i*)Array_MASK_FRAME_EDGE), (char *)line_dst);
			_mm_maskmoveu_si128(x1, _mm_load_si128((__m128i*)Array_MASK_FRAME_EDGE), (char *)(line_dst + w - 6));
		}
	}
#undef pblendbv_SSE2
#undef palignr_SSE2
}

unsigned int __stdcall RunThread(void *arg) {
	thread_t *th = (thread_t *)arg;
	WaitForSingleObject(th->he_start, INFINITE);
	while (!th->abort) {
		edgelevel_func_mt_sse2_avisynth(th);
		SetEvent(th->he_fin);
		WaitForSingleObject(th->he_start, INFINITE);
	}
	_endthreadex(0);
	return 0;
}

BOOL CreateThreads(int threads, frame_buf_t *_buf, edegelevel_prm_t *_prm, mt_control_t *mt_control) {
	if (!check_range(threads, 0, MAX_THREADS))
		return FALSE;

	BOOL ret = TRUE;
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

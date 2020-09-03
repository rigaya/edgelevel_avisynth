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

#pragma once

const int MAX_THREADS = 32;

#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))

static inline bool check_range(int value, int min, int max) {
	return (min <= value && value <= max);
}

static inline bool check_range(double value, double min, double max) {
	return (min <= value && value <= max);
}

typedef struct {
	BYTE *src_ptr[3];
	int src_pitch[3];
	int src_width[3];
	int src_height[3];
	BYTE *dst_ptr[3];
	int dst_pitch[3];
	int dst_width[3];
	int dst_height[3];
	int planes;
} frame_buf_t;

typedef struct {
	int strength;  // 特性
	int thrs;      // 閾値
	int bc;        // 黒補正
	int wc;        // 白補正
	int bit_depth; // bit深度
	int avx2;
} edegelevel_prm_t;

typedef struct {
	int thread_id;
	int threads;
	BOOL abort;
	HANDLE he_start;
	HANDLE he_fin;
	frame_buf_t *buf;
	edegelevel_prm_t prm;
} thread_t;

typedef struct {
	HANDLE th_threads[MAX_THREADS];
	HANDLE he_fin_copy[MAX_THREADS];
	thread_t th_prm[MAX_THREADS];
	int threads;
} mt_control_t;

BOOL CreateThreads(int threads, frame_buf_t *_buf, edegelevel_prm_t *_prm, mt_control_t *mt_control);
void FinishThreads(mt_control_t *mt_control);
void StartProcessFrame(mt_control_t *mt_control);

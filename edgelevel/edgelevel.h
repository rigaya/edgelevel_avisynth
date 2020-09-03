﻿#pragma once

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
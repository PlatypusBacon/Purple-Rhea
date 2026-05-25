#ifndef PROCESSING_THREAD_H
#define PROCESSING_THREAD_H
#include <stdbool.h>
#include <zephyr/kernel.h>

void processing_thread_entry(void *p1, void *p2, void *p3);
#endif

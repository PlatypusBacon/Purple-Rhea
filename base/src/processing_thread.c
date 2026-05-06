#include <stdio.h>
#include "processing_thread.h"

void processing_thread_entry(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);
    printk("Processing thread started\n");

    while (true) {
        k_sleep(K_SECONDS(1));
    }
}
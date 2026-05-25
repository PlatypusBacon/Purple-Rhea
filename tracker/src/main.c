#include <zephyr/kernel.h>
#include "processing_thread.h"

// Thread defines
#define PROC_THREAD_STACK_SIZE  8192
#define PROC_THREAD_PRIORITY    5

K_THREAD_STACK_DEFINE(proc_thread_stack, PROC_THREAD_STACK_SIZE);
static struct k_thread proc_thread_data;


int main(void)
{
    /* Give the host a moment to enumerate USB CDC-ACM before we start
     * emitting LOCATION lines, otherwise the first few are dropped. */
    k_msleep(2000);

    printk("Tracker starting\n");

    k_thread_create(&proc_thread_data,
                    proc_thread_stack,
                    K_THREAD_STACK_SIZEOF(proc_thread_stack),
                    processing_thread_entry,
                    NULL, NULL, NULL,
                    PROC_THREAD_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&proc_thread_data, "processing");
    return 0;
}
#include <gtest/gtest.h>

#include <cstddef>
#include <thread>

#include "core/thread_stack.h"

#if defined(__linux__)
#include <pthread.h>
#endif

// A thread the core starts gets glibc's 8 MiB stack on any C library: musl gives 128 KiB, and the shader compiles and
// the driver's submission paths on the core's threads run deeper than that. On glibc this holds anyway; the Alpine CI
// job is where it is tested for real.
TEST(ThreadStack, ACoreThreadGetsAtLeastEightMiB) {
#if defined(__linux__)
    mv::ensure_thread_stack();
    std::size_t size = 0;
    std::thread t([&size] {
        pthread_attr_t attr;
        if (pthread_getattr_np(pthread_self(), &attr) == 0) {
            pthread_attr_getstacksize(&attr, &size);
            pthread_attr_destroy(&attr);
        }
    });
    t.join();
    EXPECT_GE(size, std::size_t{8} << 20) << "a std::thread's stack after ensure_thread_stack(): " << size << " bytes";
#else
    GTEST_SKIP() << "Linux only";
#endif
}

#include "core/thread_stack.h"

#if defined(__linux__)
#include <pthread.h>

#include <cstddef>
#endif

namespace mv {

void ensure_thread_stack() {
#if defined(__linux__)
    constexpr std::size_t kStack = std::size_t{8} << 20;
    pthread_attr_t attr;
    if (pthread_getattr_default_np(&attr) != 0) return;
    std::size_t size = 0;
    if (pthread_attr_getstacksize(&attr, &size) == 0 && size < kStack && pthread_attr_setstacksize(&attr, kStack) == 0)
        pthread_setattr_default_np(&attr);
    pthread_attr_destroy(&attr);
#endif
}

}  // namespace mv

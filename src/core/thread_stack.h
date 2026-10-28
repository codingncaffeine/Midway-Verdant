#pragma once

namespace mv {

// Threads get the stack glibc gives them (8 MiB) wherever the core runs. musl (Alpine) gives a new thread 128 KiB, and
// the core's threads run deep work: glslang and the GPU driver's own shader compiler on the compile workers, the
// driver's submission paths on the CPU and FIFO threads, 30-64 KiB frames of the core's own. Called before the core
// starts any thread (retro_init, the shell's main); it only ever raises the process's default, never lowers it.
void ensure_thread_stack();

}  // namespace mv

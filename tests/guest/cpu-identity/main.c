// What the processor reports of itself, printed on the screen: CPUID leaves 0-3 and 0x80000000, and the model-specific
// registers a title can read to learn the bus speed, clock ratio, microcode and platform. A guest program for the
// emulator's processor identity (built with nxdk; run as a disc and read the picture).
//
// On the fast core and the cycle-counted core alike the console's processor reads:
//   cpuid 0: 00000002 756e6547 6c65746e 49656e69     ("GenuineIntel", highest leaf 2)
//   cpuid 1: 0000068a 00000000 00000000 0383f9ff     (family 6 model 8 stepping 10; MMX, SSE, no SSE2)
//   cpuid 2: 03020101 00000000 00000000 0c040841     (the 128 KiB second-level cache)
//   msr 02a: 00000000c5040000                        (133 MHz bus, clock ratio 5.5)
// A read of a register the processor lacks would stop the kernel with a general protection fault, so only registers
// of the P6 family are read.
#include <hal/debug.h>
#include <hal/video.h>
#include <windows.h>

static unsigned long long rdmsr(unsigned index)
{
    unsigned lo, hi;
    __asm__ volatile("rdmsr" : "=a"(lo), "=d"(hi) : "c"(index));
    return ((unsigned long long)hi << 32) | lo;
}

static void cpuid(unsigned leaf, unsigned r[4])
{
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(0));
}

int main(void)
{
    static const unsigned msrs[] = {0x17, 0x2A, 0x8B, 0xC1, 0x186, 0xFE, 0x119, 0x11E, 0x179, 0x1E0, 0x2FF};
    unsigned r[4];
    XVideoSetMode(640, 480, 32, REFRESH_DEFAULT);
    for (unsigned leaf = 0; leaf <= 3; ++leaf) {
        cpuid(leaf, r);
        debugPrint("cpuid %u: %08x %08x %08x %08x\n", leaf, r[0], r[1], r[2], r[3]);
    }
    cpuid(0x80000000u, r);
    debugPrint("cpuid 80000000: %08x %08x %08x %08x\n", r[0], r[1], r[2], r[3]);
    for (unsigned i = 0; i < sizeof msrs / sizeof msrs[0]; ++i) {
        unsigned long long v = rdmsr(msrs[i]);
        debugPrint("msr %03x: %08x%08x\n", msrs[i], (unsigned)(v >> 32), (unsigned)v);
    }
    debugPrint("done\n");
    while (1) Sleep(2000);
    return 0;
}

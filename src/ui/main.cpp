#include <cstdio>
#include <cstring>

#include "core/version.h"

namespace {

void print_usage() {
    std::printf("%s %s\n\n", mv::product_name().data(), mv::version_string().data());
    std::printf("Usage: midway-verdant [--version] [--help]\n\n"
                "A low-level emulator of the original Xbox. The machine is not here yet;\n"
                "this build only answers --version.\n");
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("%s %s\n", mv::product_name().data(), mv::version_string().data());
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
        std::fprintf(stderr, "midway-verdant: unknown option %s\n", argv[i]);
        return 2;
    }
    print_usage();
    return 2;
}

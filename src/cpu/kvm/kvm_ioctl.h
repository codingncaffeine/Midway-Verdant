#pragma once
#include <sys/ioctl.h>

namespace mv {

// ioctl's request parameter is unsigned long in glibc and int in musl (Alpine): KVM's request numbers (KVM_GET_REGS is
// 0x8090AE81) do not fit an int, and -Wsign-conversion stopped the build there. The request goes in as whatever type
// this C library's ioctl declares, read from its own signature; the kernel takes the bits back as unsigned either way.
namespace detail {
template <class R, class Fd, class Request>
Request ioctl_request_type(R (*)(Fd, Request, ...));
}
using IoctlRequest = decltype(detail::ioctl_request_type(&::ioctl));

template <class... Args>
inline int kvm_ioctl(int fd, unsigned long request, Args... args) {
    return ::ioctl(fd, static_cast<IoctlRequest>(request), args...);
}

}  // namespace mv

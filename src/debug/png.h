#pragma once
#include <filesystem>
#include <string>

#include "core/types.h"

namespace mv::png {

// Writes RGBA8 pixels (row stride in bytes) as an uncompressed PNG; false with a reason on failure.
bool write(const std::filesystem::path& path, u32 width, u32 height, const u8* rgba, u32 stride, std::string& error);

}  // namespace mv::png

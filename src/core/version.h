#pragma once
#include <string_view>

namespace mv {

// "Midway Verdant X.Y.Z": the line --version prints and every log begins with.
std::string_view product_name();
std::string_view version_string();

}  // namespace mv

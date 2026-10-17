#include "core/version.h"

#include "mv/version.h"

namespace mv {

std::string_view product_name() { return "Midway Verdant"; }
std::string_view version_string() { return MV_VERSION_STRING; }

}  // namespace mv

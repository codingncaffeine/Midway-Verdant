#include <gtest/gtest.h>

#include <regex>
#include <string>

#include "core/version.h"

TEST(Version, ProductNameAndSemanticVersion) {
    EXPECT_EQ(mv::product_name(), "Midway Verdant");
    const std::string v{mv::version_string()};
    EXPECT_TRUE(std::regex_match(v, std::regex{R"(\d+\.\d+\.\d+)"})) << v;
}

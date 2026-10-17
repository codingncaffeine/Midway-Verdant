#pragma once
#include <string>
#include <vector>

#include "core/types.h"

// GLSL to SPIR-V through glslang, for the shaders the translators emit. Thread-safe after the first call.
namespace mv::pgraph {

enum class ShaderStage { Vertex, Fragment };

struct CompiledShader {
    std::vector<u32> spirv;
    std::string log;  // errors and warnings, empty on a clean compile
    bool ok() const { return !spirv.empty(); }
};

CompiledShader compile_glsl(ShaderStage stage, const std::string& source, const std::string& name);

}  // namespace mv::pgraph

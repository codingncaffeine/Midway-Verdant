#include "nv2a/shaders/glsl_compiler.h"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <mutex>

namespace mv::pgraph {

namespace {
std::once_flag g_init;
}

CompiledShader compile_glsl(ShaderStage stage, const std::string& source, const std::string& name) {
    std::call_once(g_init, [] { glslang::InitializeProcess(); });
    CompiledShader out;
    const EShLanguage lang = stage == ShaderStage::Vertex ? EShLangVertex : EShLangFragment;
    glslang::TShader shader(lang);
    const char* src = source.c_str();
    const char* names = name.c_str();
    shader.setStringsWithLengthsAndNames(&src, nullptr, &names, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    const EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 100, false, messages)) {
        out.log = std::string(shader.getInfoLog()) + shader.getInfoDebugLog();
        return out;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        out.log = std::string(program.getInfoLog()) + program.getInfoDebugLog();
        return out;
    }
    spv::SpvBuildLogger logger;
    glslang::SpvOptions options;
    options.disableOptimizer = true;
    options.validate = true;
    std::vector<unsigned int> words;
    glslang::GlslangToSpv(*program.getIntermediate(lang), words, &logger, &options);
    out.spirv.assign(words.begin(), words.end());
    out.log = logger.getAllMessages();
    if (shader.getInfoLog() && *shader.getInfoLog()) out.log += shader.getInfoLog();
    return out;
}

}  // namespace mv::pgraph

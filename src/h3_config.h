// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

namespace h3image
{
inline constexpr int kLatentChannels = 24;
inline constexpr int kLatentScale = 16;
inline constexpr int kTransformerPatch = 2;
inline constexpr int kPackedWidth = kLatentChannels * kTransformerPatch * kTransformerPatch;
inline constexpr int kTextWidth = 5120;
inline constexpr int kTransformerWidth = 5376;
inline constexpr int kTransformerLayers = 50;
inline constexpr int kModulationWidth = 6 * 3 * kTransformerWidth;
inline constexpr int kVisionPatch = 16;
inline constexpr int kVisionMerge = 2;
inline constexpr int kVisionPixelWidth = 3 * 2 * kVisionPatch * kVisionPatch;
inline constexpr int kMaxTextTokens = 4096;
inline constexpr float kReferenceNoiseAug = 0.999f;

// Parameters of the MiniMax-H3-Image BF16 deployment.
inline constexpr std::array<float, 24> kLatentMean = { 0.85809034109115601f, -0.96065914630889893f, 1.0661640167236328f,
    -0.50903254747390747f, -0.27275818586349487f, -1.3675414323806763f, -0.25532549619674683f, -0.26907554268836975f,
    -0.53768408298492432f, -0.046409729868173599f, 0.66573703289031982f, 0.19690127670764923f, -0.54606080055236816f,
    -0.40353420376777649f, -0.23683024942874908f, 0.25928452610969543f, -0.30133944749832153f, 0.21134199202060699f,
    -1.1206848621368408f, 0.35819333791732788f, -0.042251437902450562f, 0.26048299670219421f, 0.22864092886447906f,
    0.70560318231582642f };
inline constexpr std::array<float, 24> kLatentStd
    = { 1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f, 1.5636216402053833f,
          2.194143533706665f, 0.96531379222869873f, 1.0569885969161987f, 0.84194892644882202f, 0.77299529314041138f,
          1.8955937623977661f, 0.94684183597564697f, 0.79968094825744629f, 0.44988900423049927f, 0.71973997354507446f,
          0.69362932443618774f, 2.9610950946807861f, 2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
          3.276226282119751f, 3.1627357006072998f, 2.2816812992095947f, 2.6127843856811523f };
inline constexpr std::array<float, 7> kSampleSigmas
    = { 1.0f, 0.96752369403839111f, 0.88754850625991821f, 0.75f, 0.53277391195297241f, 0.23200887441635132f, 0.0f };
}

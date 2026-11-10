/* Copyright (C) 2021 Edgar B
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#pragma once

#include "btu/tex/detail/common.hpp"
#include "btu/tex/dimension.hpp"
#include "btu/tex/formats.hpp"
#include "compression_device.hpp"

#include <btu/common/games.hpp>
#include <btu/common/json.hpp>

#include <variant>

namespace btu::tex {
class Texture;
class CrunchTexture;

struct Settings
{
    [[nodiscard]] static auto get(Game game) noexcept -> const Settings &;

    Game game = Game::SSE;

    bool compress = false;
    bool force_crunch = false; // Use crnlib's DXT encoder (via CrunchTexture path) instead of DirectXTex

    // When true, restricts `compress` to textures that aren't already block-compressed: an
    // already-compressed source (DirectXTex IsCompressed(), or CrunchTexture::is_packed()) is left
    // on its current format entirely instead of being normalized towards output_format.compressed
    // /compressed_without_alpha. See best_output_format() in optimize.cpp.
    bool compress_uncompressed_only = false;

    // Opt-in: store diffuse DXT5 whose alpha is exactly 0/255 everywhere as DXT1 with 1-bit alpha.
    // Only when it would be the sole step; see compute_optimization_steps().
    bool bc3_to_bc1a = false;

    std::variant<std::monostate, util::ResizeRatio, Dimension> resize;
    bool mipmaps = false;

    bool use_format_whitelist = false;
    std::vector<DXGI_FORMAT> allowed_formats;

    BestFormatFor output_format;

    std::vector<std::u8string> landscape_textures;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Settings,
                                   game,
                                   compress,
                                   force_crunch,
                                   compress_uncompressed_only,
                                   bc3_to_bc1a,
                                   resize,
                                   mipmaps,
                                   use_format_whitelist,
                                   allowed_formats,
                                   output_format,
                                   landscape_textures)

struct OptimizationSteps
{
    std::optional<Dimension> resize;
    bool add_transparent_alpha = false;
    bool mipmaps               = false;
    DXGI_FORMAT best_format    = DXGI_FORMAT_UNKNOWN;
    bool convert               = false;
    // DXT5/BC3 whose alpha is fully opaque, re-stored as DXT1/BC1 (half the size). When it is the
    // only step, optimize() copies the colour blocks instead of re-encoding, so it is lossless.
    bool bc3_to_bc1 = false;
    // DXT5/BC3 diffuse whose alpha is exactly 0/255 everywhere, re-stored as DXT1 with 1-bit alpha.
    bool bc3_to_bc1a = false;

    auto operator<=>(const OptimizationSteps &) const noexcept = default;
};

[[nodiscard]] auto optimize(Texture &&file, OptimizationSteps sets, CompressionDevice &dev) noexcept -> Result;
[[nodiscard]] auto optimize(CrunchTexture &&file,
                            OptimizationSteps sets,
                            CompressionDevice &dev) noexcept -> ResultCrunch;
[[nodiscard]] auto compute_optimization_steps(const Texture &file,
                                              const Settings &sets) noexcept -> OptimizationSteps;
[[nodiscard]] auto compute_optimization_steps(const CrunchTexture &file,
                                              const Settings &sets) noexcept -> OptimizationSteps;
} // namespace btu::tex

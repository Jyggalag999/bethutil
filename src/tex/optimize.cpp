/* Copyright (C) 2021 Edgar B
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#include "btu/tex/optimize.hpp"

#include "btu/tex/compression_device.hpp"
#include "btu/tex/crunch_functions.hpp"
#include "btu/tex/functions.hpp"

#include <btu/common/algorithms.hpp>
#include <btu/common/metaprogramming.hpp>
#include <btu/tex/dxtex.hpp>

#include <crunch/crn_image_utils.h>

namespace btu::tex {
auto optimize(Texture &&file, OptimizationSteps sets, CompressionDevice &dev) noexcept -> Result
{
    const auto &info = file.get().GetMetadata();
    // All operations require a decompressed texture.
    const auto must_decompress = DirectX::IsCompressed(info.format);
    // Special case - force conversion if result shouldn't have alpha to get rid of alpha bits that are added by DirectX.
    const auto should_convert = sets.convert || must_decompress || !DirectX::HasAlpha(sets.best_format);
    auto res                  = Result{std::move(file)};

    if (must_decompress)
        res = std::move(res).and_then(decompress);
    if (sets.resize)
        res = std::move(res).and_then(
            [&](Texture &&tex) { return resize(std::move(tex), sets.resize.value()); });
    if (sets.add_transparent_alpha)
        res = std::move(res).and_then(make_transparent_alpha);
    if (sets.mipmaps)
        res = std::move(res).and_then(BTU_RESOLVE_OVERLOAD(generate_mipmaps));

    // We have uncompressed the texture. If it was compressed, it's best to convert it to a better format
    const auto cur_format_is_same_as_best = res && res->get().GetMetadata().format == sets.best_format;

    if (should_convert && !cur_format_is_same_as_best)
    {
        auto out = sets.best_format;
        res      = std::move(res)
                  // // safety check: make sure we don't remove the alpha
                  .and_then([&](Texture &&tex) -> Result {
                      if (!DirectX::HasAlpha(out) && !tex.get().IsAlphaAllOpaque())
                          return tl::make_unexpected(Error(TextureErr::BadInput));
                      return std::move(tex);
                  })
                  .and_then([&](Texture &&tex) { return convert(std::move(tex), out, dev); });
    }

    return res;
}

auto optimize(CrunchTexture &&file,
              OptimizationSteps sets,
              [[maybe_unused]] CompressionDevice &dev) noexcept -> ResultCrunch
{
    const auto must_decompress = file.get().is_packed() && (sets.resize || sets.mipmaps);
    const auto should_convert  = sets.convert || must_decompress;

    auto res = ResultCrunch{std::move(file)};

    if (sets.resize)
        res = std::move(res).and_then(
            [&](CrunchTexture &&tex) { return resize(std::move(tex), sets.resize.value()); });
    if (sets.mipmaps)
        res = std::move(res).and_then(BTU_RESOLVE_OVERLOAD(generate_mipmaps));
    if (should_convert)
        res = std::move(res).and_then(
            [&](CrunchTexture &&tex) { return convert(std::move(tex), sets.best_format); });

    return res;
}

/// SSE landscape textures uses alpha channel as specularity
/// Textures with opaque alpha are thus rendered shiny
/// To fix this, alpha has to be made transparent
[[nodiscard]] static auto transparent_alpha_required(const Texture &file, const Settings &sets) -> bool
{
    const auto &tex         = file.get();
    const auto path         = canonize_path(file.get_load_path());
    const bool is_landscape = common::contains(sets.landscape_textures, path);
    if (!is_landscape)
        return false;

    return tex.IsAlphaAllOpaque();
}

[[nodiscard]] static auto is_bad_cubemap(const TexMetadata &info) noexcept -> bool
{
    const bool is_cubemap   = info.IsCubemap();
    const bool uncompressed = !DirectX::IsCompressed(info.format);

    const bool opaque_alpha = info.GetAlphaMode() == DirectX::TEX_ALPHA_MODE::TEX_ALPHA_MODE_OPAQUE;
    const bool no_alpha     = !DirectX::HasAlpha(info.format);
    const bool bad_alpha    = opaque_alpha || no_alpha;

    return is_cubemap && uncompressed && bad_alpha;
}

[[nodiscard]] static auto has_opaque_alpha(const ScratchImage &tex) noexcept -> bool
{
    using enum DirectX::TEX_ALPHA_MODE;

    const auto has_alpha  = DirectX::HasAlpha(tex.GetMetadata().format);
    const auto alpha_mode = tex.GetMetadata().GetAlphaMode();
    return !has_alpha || alpha_mode == TEX_ALPHA_MODE_OPAQUE || tex.IsAlphaAllOpaque();
}

/// crnlib's own mipmapped_texture::has_alpha() is a format-level check only ("does this pixel
/// format carry an alpha channel"), not a pixel-level one - unlike DirectXTex's IsAlphaAllOpaque()
/// above, it says nothing about whether that channel's values are actually all opaque. Left
/// unchecked, that meant diffuse textures with a fully-opaque alpha channel always got routed to
/// an alpha-preserving compressed format (e.g. BC3) via the crunch path even when DirectXTex would
/// correctly pick the smaller alpha-less one (e.g. BC1) for the exact same texture. This mirrors
/// has_opaque_alpha(ScratchImage) above using crnlib's own real per-pixel scan
/// (image_utils::has_alpha, which is true iff some pixel's alpha < 255) as the equivalent of
/// IsAlphaAllOpaque().
[[nodiscard]] static auto has_opaque_alpha(const CrunchTexture &file) noexcept -> bool
{
    const auto &tex = file.get();

    // Fast path: format itself carries no alpha channel at all, nothing to scan.
    if (!tex.has_alpha())
        return true;

    crnlib::image_u8 img;
    const auto *unpacked = tex.get_level_image(0, 0, img); // auto-decompresses if currently packed
    if (unpacked == nullptr)
        return false; // failed to unpack: be conservative, assume alpha might matter

    return !crnlib::image_utils::has_alpha(*unpacked);
}

template<class Tex>
[[nodiscard]] static auto can_be_compressed(const Tex &file) noexcept -> bool
{
    const Dimension dim  = file.get_dimension();
    const bool too_small = dim.w < 4 || dim.h < 4;
    const bool pow2      = util::is_pow2(dim.w) && util::is_pow2(dim.h);

    return !too_small && pow2;
}

template<class Tex>
[[nodiscard]] static auto is_tga(const Tex &file) noexcept -> bool
{
    try
    {
        const auto ext = file.get_load_path().extension();
        return str_compare(ext.u8string(), u8".tga", common::CaseSensitive::No);
    }
    catch (const std::exception &)
    {
        return false;
    }
}

[[nodiscard]] static auto conversion_required(const Texture &file, const Settings &sets) noexcept -> bool
{
    const auto &info = file.get().GetMetadata();

    const bool forbidden_format = sets.use_format_whitelist
                                  && !common::contains(sets.allowed_formats, info.format);

    const bool is_bad_cube = is_bad_cubemap(info);
    const bool bad         = forbidden_format || is_bad_cube;

    return bad || is_tga(file);
}

[[nodiscard]] static auto target_dimensions(const Dimension dim,
                                            const Settings &sets) -> std::optional<Dimension>
{
    return std::visit(common::Overload{
                          [](std::monostate) -> std::optional<Dimension> { return {}; },
                          [&](const util::ResizeRatio &r) {
                              return std::optional(compute_resize_dimension(dim, r));
                          },
                          [&](const Dimension target) {
                              return std::optional(util::compute_resize_dimension(dim, target));
                          },
                      },
                      sets.resize);
}

[[nodiscard]] static auto best_output_format(const Texture &file,
                                             const Settings &sets,
                                             const bool force_alpha) noexcept -> DXGI_FORMAT
{
    const auto &tex  = file.get();
    const auto &info = tex.GetMetadata();

    // "Only compress what's uncompressed": leave an already block-compressed source exactly as it
    // is instead of normalizing it towards output_format.compressed/compressed_without_alpha.
    if (sets.compress_uncompressed_only && DirectX::IsCompressed(info.format))
        return info.format;

    return guess_best_format(info.format,
                             sets.output_format,
                             GuessBestFormatArgs{.opaque_alpha     = has_opaque_alpha(tex),
                                                 .allow_compressed = sets.compress && can_be_compressed(file),
                                                 .force_alpha      = force_alpha});
}

[[nodiscard]] static auto best_output_format(const CrunchTexture &file,
                                             const Settings &sets,
                                             bool force_alpha) noexcept -> DXGI_FORMAT
{
    // Same "only compress what's uncompressed" carve-out as the DirectXTex path above, using
    // crnlib's own is_packed() (true iff the texture is currently DXT-encoded) as the equivalent
    // of DirectX::IsCompressed() - but only when crnlib can actually re-encode into that exact
    // source format. crnlib's encoder repertoire is far smaller than its decoder one (notably no
    // BC7), so for a source format it has no encoder for, "leave it as-is" isn't achievable once
    // a resize/mipmap regen forces a decompress+recompress cycle (see must_decompress in
    // optimize(CrunchTexture&&, ...) below) - that used to just fail the whole file with BadInput.
    // Recompress into this profile's configured compressed format instead: guess_best_format
    // treats "already compressed" as reason enough to pick formats.compressed/
    // compressed_without_alpha regardless of sets.compress, and can_use_crunch (main_process.cpp)
    // already guarantees those two are crunch-encodable whenever force_crunch is on.
    if (sets.compress_uncompressed_only && file.get().is_packed()
        && crunch_supports_format(file.get_format_as_dxgi()))
        return file.get_format_as_dxgi();

    return guess_best_format(file.get_format_as_dxgi(),
                             sets.output_format,
                             GuessBestFormatArgs{.opaque_alpha     = has_opaque_alpha(file),
                                                 .allow_compressed = sets.compress
                                                                     && can_be_compressed<CrunchTexture>(file),
                                                 .force_alpha = force_alpha});
}

auto compute_optimization_steps(const Texture &file, const Settings &sets) noexcept -> OptimizationSteps
{
    const auto &tex  = file.get();
    const auto &info = tex.GetMetadata();

    auto res = OptimizationSteps{};

    // Check if conversion is a must.
    res.convert = conversion_required(file, sets);

    // Do not compress the image if already compressed.
    if (sets.compress && !DirectX::IsCompressed(info.format))
        res.convert = true;

    const auto dim = Dimension{.w = info.width, .h = info.height};

    const std::optional<Dimension> target_dim = target_dimensions(dim, sets);
    if (target_dim.has_value() && dim != target_dim.value())
        res.resize = target_dim.value();

    if (sets.game == Game::SSE)
        if (transparent_alpha_required(file, sets))
            res.add_transparent_alpha = true;

    const bool opt_mip = optimal_mip_count(file.get_dimension()) == info.mipLevels;
    if ((sets.mipmaps && !opt_mip)
        || (info.mipLevels > 1 && res.resize)) // resize removes mips if there are any, regenerate them
        res.mipmaps = true;

    // I prefer to keep steps independent, but this one has to depend on add_transparent_alpha. If we add an alpha, the output format must have alpha
    res.best_format = best_output_format(file, sets, res.add_transparent_alpha);

    return res;
}

auto compute_optimization_steps(const CrunchTexture &file, const Settings &sets) noexcept -> OptimizationSteps
{
    const auto &tex = file.get();

    auto res = OptimizationSteps{};

    // Resizing.
    const auto dim = file.get_dimension();

    const std::optional<Dimension> target_dim = target_dimensions(dim, sets);
    if (target_dim.has_value() && dim != target_dim.value())
        res.resize = target_dim.value();

    // Mipmaps. Same "already has the right chain, don't touch it" guard as the DirectXTex path
    // above (opt_mip) - without it, the global mipmaps toggle alone forced every already-optimal
    // crunch-path texture through a needless (and lossy, since it also recompresses) decompress/
    // regenerate/recompress round-trip for a mip chain that was already correct.
    const bool opt_mip = optimal_mip_count(file.get_dimension()) == tex.get_num_levels();
    if ((sets.mipmaps && !opt_mip)
        || (tex.get_num_levels() > 1 && res.resize)) // resize removes mips if there are any, regenerate them
        res.mipmaps = true;

    // Conversion.
    // Check if conversion is a must.
    res.convert = is_tga<CrunchTexture>(file);

    // Do not compress the image if already compressed.
    if (sets.compress && !tex.is_packed())
        res.convert = true;

    res.best_format = best_output_format(file, sets, /*force_alpha=*/false);

    return res;
}

// NOLINTNEXTLINE(misc-no-recursion)
auto Settings::get(const Game game) noexcept -> const Settings &
{
    static auto tes3_sets = [&] {
        return Settings{
            .game = Game::TES3,
            .compress = false,
            .resize = {},
            .mipmaps = true,
            .use_format_whitelist = true,
            .allowed_formats = {
                DXGI_FORMAT_BC3_UNORM,
                DXGI_FORMAT_BC1_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_B8G8R8X8_UNORM,
            }, // Older titles only support DXT1-DXT5 compression.
            .output_format = {.uncompressed = DXGI_FORMAT_B8G8R8A8_UNORM,
                              .uncompressed_without_alpha = DXGI_FORMAT_B8G8R8X8_UNORM,
                              .compressed = DXGI_FORMAT_BC3_UNORM,
                              .compressed_without_alpha = DXGI_FORMAT_BC1_UNORM},
            .landscape_textures = {}, // Unknown
        };
    }();

    switch (game)
    {
        case Game::TES3: return tes3_sets;
        case Game::TES4:
        {
            static auto tes4_sets = [&] {
                auto sets = tes3_sets;
                sets.game = Game::TES4;
                return sets;
            }();
            return tes4_sets;
        }
        case Game::FNV:
        {
            static auto fnv_sets = [&] {
                auto sets = tes3_sets;
                sets.game = Game::FNV;
                return sets;
            }();
            return fnv_sets;
        }
        case Game::SLE:
        {
            static auto sle_sets = [&] {
                auto sets = tes3_sets;
                sets.game = Game::SLE;
                return sets;
            }();
            return sle_sets;
        }
        case Game::SSE:
        {
            static auto sse_sets = [&] {
                auto sets            = tes3_sets;
                sets.compress        = true;
                sets.allowed_formats = {
                    DXGI_FORMAT_BC7_UNORM,
                    DXGI_FORMAT_BC5_UNORM,
                    DXGI_FORMAT_BC3_UNORM,
                    DXGI_FORMAT_BC1_UNORM,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_FORMAT_B8G8R8X8_UNORM,
                };
                sets.output_format.compressed               = DXGI_FORMAT_BC7_UNORM;
                sets.output_format.compressed_without_alpha = DXGI_FORMAT_BC7_UNORM;
                sets.game                                   = Game::SSE;
                return sets;
            }();
            return sse_sets;
        }
        case Game::FO4:
        {
            // NOLINTNEXTLINE(misc-no-recursion)
            static auto fo4_sets = [&] {
                auto sets = get(Game::SSE);
                sets.game = Game::FO4;
                return sets;
            }();
            return fo4_sets;
        }
        case Game::Starfield:
        {
            // NOLINTNEXTLINE(misc-no-recursion)
            static auto starfield_sets = [&] {
                auto sets = get(Game::SSE);
                sets.game = Game::Starfield;
                return sets;
            }();
            return starfield_sets;
        }
        case Game::Custom: return tes3_sets;
    }
    return tes3_sets;
}
} // namespace btu::tex

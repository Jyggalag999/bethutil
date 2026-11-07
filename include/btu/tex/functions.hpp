#pragma once

#include "btu/tex/detail/common.hpp"
#include "btu/tex/detail/formats_string.hpp"
#include "btu/tex/dimension.hpp"
#include "btu/tex/texture.hpp"

#include <btu/tex/compression_device.hpp>

namespace btu::tex {
[[nodiscard]] auto decompress(Texture &&file) -> Result;
[[nodiscard]] auto make_transparent_alpha(Texture &&file) -> Result;
[[nodiscard]] auto convert(Texture &&file, DXGI_FORMAT format, CompressionDevice &dev) -> Result;

[[nodiscard]] constexpr auto optimal_mip_count(Dimension dim) noexcept -> size_t
{
    size_t mips = 1;
    auto size   = std::max(dim.w, dim.h);

    while (size > 1)
    {
        size /= 2;
        ++mips;
    }
    return mips;
}

[[nodiscard]] auto generate_mipmaps(Texture &&file) -> Result;

/// The BC1 format with the same colour space as `bc3` (BC3_UNORM -> BC1_UNORM, BC3_UNORM_SRGB ->
/// BC1_UNORM_SRGB), or DXGI_FORMAT_UNKNOWN when `bc3` is not a BC3 format.
[[nodiscard]] auto bc1_counterpart(DXGI_FORMAT bc3) noexcept -> DXGI_FORMAT;
/// True when `tex` is BC3 and every texel of every image decodes to alpha 255. Reads the alpha
/// blocks directly (no decompression) and is conservative: a texel whose value depends on decoder
/// rounding counts as not opaque.
[[nodiscard]] auto bc3_alpha_all_opaque(const ScratchImage &tex) noexcept -> bool;
/// Re-stores a BC3 texture as BC1 by copying each block's colour half, swapping endpoints where
/// BC1 would otherwise read the block in its 3-colour mode. Bit-exact colours, alpha dropped, so
/// only call it when bc3_alpha_all_opaque() is true.
[[nodiscard]] auto bc3_to_bc1(Texture &&file) -> Result;
[[nodiscard]] auto resize(Texture &&file, Dimension dim) -> Result;
} // namespace btu::tex

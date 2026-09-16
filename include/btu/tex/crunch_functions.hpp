#pragma once

#include "btu/tex/crunch_texture.hpp"
#include "btu/tex/detail/common.hpp"
#include "btu/tex/detail/formats_string.hpp"
#include "btu/tex/dimension.hpp"

#include <crunch/crn_dxt_image.h>
#include <crunch/crn_texture_conversion.h>
#include <crunch/dds_defs.h>

namespace btu::tex {
using crnlib::texture_conversion::convert_params;
[[nodiscard]] auto resize(CrunchTexture &&file, Dimension dim) -> ResultCrunch;
[[nodiscard]] auto generate_mipmaps(CrunchTexture &&file) -> ResultCrunch;
[[nodiscard]] auto convert(CrunchTexture &&file, DXGI_FORMAT format) -> ResultCrunch;

/// crnlib (the library backing the CrunchTexture path) can only pack into the handful of formats
/// listed in convert()'s switch below - notably it has no BC7 encoder. Callers deciding whether to
/// route a texture through the crunch pipeline at all should check this first: if the settings ask
/// for a format crnlib can't produce (e.g. BC7, used by SSE/FO4/Starfield), convert() would just
/// fail and the texture would be left unprocessed. Prefer the DirectXTex path (btu::tex::convert for
/// Texture, which does support BC7 via bc7enc) for those formats instead.
[[nodiscard]] auto crunch_supports_format(DXGI_FORMAT format) noexcept -> bool;
} // namespace btu::tex

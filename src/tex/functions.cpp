#include <btu/tex/compression_device.hpp>
#include <btu/tex/dxtex.hpp>
#include <btu/tex/error_code.hpp>
#include <btu/tex/functions.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace btu::tex {
auto decompress(Texture &&file) -> Result
{
    const auto &tex    = file.get();
    const auto &img    = tex.GetImages();
    const size_t &nimg = tex.GetImageCount();
    const auto &info   = tex.GetMetadata();

    ScratchImage timage;
    const auto hr = Decompress(img, nimg, info, DXGI_FORMAT_UNKNOWN /* picks good default */, timage);
    if (FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    file.set(std::move(timage));
    return std::move(file);
}

auto make_transparent_alpha(Texture &&file) -> Result
{
    const auto &tex = file.get();

    if (!DirectX::HasAlpha(tex.GetMetadata().format))
        return tl::make_unexpected(Error(TextureErr::BadInput));

    constexpr auto transform = [](DirectX::XMVECTOR *out_pixels,
                                  const DirectX::XMVECTOR *in_pixels,
                                  const size_t width,
                                  [[maybe_unused]] size_t) {
        const auto transparent = DirectX::XMVectorSet(0, 0, 0, 0);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        const auto *end = in_pixels + width;
        std::transform(in_pixels, end, out_pixels, [&](auto pix) {
            return XMVectorSelect(transparent, pix, DirectX::g_XMSelect1110);
        });
    };

    ScratchImage timage;
    const auto hr = TransformImage(tex.GetImages(), tex.GetImageCount(), tex.GetMetadata(), transform, timage);

    if (FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    file.set(std::move(timage));
    return std::move(file);
}

static auto convert_uncompressed(const ScratchImage &image,
                                 ScratchImage &timage,
                                 DXGI_FORMAT format,
                                 [[maybe_unused]] CompressionDevice &dummy) -> HRESULT
{
    const auto *const img = image.GetImages();
    if (img == nullptr)
        return E_INVALIDARG;
    const size_t nimg = image.GetImageCount();

    return Convert(img,
                   nimg,
                   image.GetMetadata(),
                   format,
                   DirectX::TEX_FILTER_SEPARATE_ALPHA,
                   DirectX::TEX_THRESHOLD_DEFAULT,
                   timage);
}

// The reason for this to be separated in another file is that both libraries define DXGI_FORMAT
// So there's a conflict
// Also, we use another library because DirectXTex BC7 CPU encoder is unbearably slow
auto convert_bc7(const uint8_t *source, uint8_t *dest, uint32_t width, uint32_t height, size_t slice_pitch)
    -> tl::expected<void, Error>;

static auto convert_compressed(const ScratchImage &image,
                               ScratchImage &timage,
                               DXGI_FORMAT format,
                               [[maybe_unused]] CompressionDevice &dev) -> HRESULT
{
    const auto *const img = image.GetImages();
    if (img == nullptr)
        return E_INVALIDARG;
    const size_t nimg = image.GetImageCount();

    const bool bc6hbc7 = [&]() noexcept {
        switch (format)
        {
            case DXGI_FORMAT_BC7_TYPELESS:
            case DXGI_FORMAT_BC7_UNORM:
            case DXGI_FORMAT_BC7_UNORM_SRGB: return true;
            default: return false;
        }
    }();

    if (bc6hbc7)
    {
        // hardware impl, only works on d3d11
#if defined(__d3d11_h__) || defined(__d3d11_x_h__)
        HRESULT bc7_ret    = E_FAIL;
        const bool applied = dev.try_apply([&](auto *d3d11_dev) {
            bc7_ret = Compress(d3d11_dev,
                               img,
                               nimg,
                               image.GetMetadata(),
                               format,
                               DirectX::TEX_COMPRESS_DEFAULT,
                               DirectX::TEX_THRESHOLD_DEFAULT,
                               timage);
        });

        if (applied)
            return bc7_ret;
#endif

        // slower bc7 impl, but faster than dxtex
        auto metadata   = image.GetMetadata();
        metadata.format = DXGI_FORMAT_BC7_UNORM;
        const auto hr   = timage.Initialize(metadata);
        if (FAILED(hr))
            return hr;

        for (size_t i = 0; i < image.GetImageCount(); ++i)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            const auto &simg = image.GetImages()[i];
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            const auto &timg = timage.GetImages()[i];

            const auto res = convert_bc7(simg.pixels,
                                         timg.pixels,
                                         static_cast<uint32_t>(simg.width),
                                         static_cast<uint32_t>(simg.height),
                                         simg.slicePitch);
            if (!res)
                return E_FAIL;
        }
        return S_OK;
    }

    constexpr auto compress_flags = [] {
#if _OPENMP
        return DirectX::TEX_COMPRESS_PARALLEL;
#else
        return DirectX::TEX_COMPRESS_DEFAULT;
#endif
    }();

    return Compress(img,
                    nimg,
                    image.GetMetadata(),
                    format,
                    compress_flags,
                    DirectX::TEX_THRESHOLD_DEFAULT,
                    timage);
}

auto convert(Texture &&file, DXGI_FORMAT format, CompressionDevice &dev) -> Result
{
    const auto &tex = file.get();
    const auto info = tex.GetMetadata();

    const bool uncompressed_required = info.width < 4 || info.height < 4;
    //Textures smaller than that cannot be compressed
    if (uncompressed_required && DirectX::IsCompressed(info.format))
        return tl::make_unexpected(Error(TextureErr::BadInput));

    ScratchImage timage;

    const auto f = DirectX::IsCompressed(format) ? convert_compressed : convert_uncompressed;

    if (const auto hr = f(tex, timage, format, dev); FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    file.set(std::move(timage));
    return std::move(file);
}

static auto prepare_generate_mipmaps(Texture &&file) -> Result
{
    const auto &tex = file.get();
    // Mips generation only works on a single base image, so strip off existing mip levels
    const auto &info = tex.GetMetadata();
    ScratchImage timage;

    TexMetadata mdata = info;
    mdata.mipLevels   = 1;
    if (const auto hr = timage.Initialize(mdata); FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    for (size_t i = 0; i < info.arraySize; ++i)
    {
        const auto hr = CopyRectangle(*tex.GetImage(0, i, 0),
                                      DirectX::Rect(0, 0, info.width, info.height),
                                      *timage.GetImage(0, i, 0),
                                      DirectX::TEX_FILTER_SEPARATE_ALPHA,
                                      0,
                                      0);
        if (FAILED(hr))
            return tl::make_unexpected(error_from_hresult(hr));
    }

    file.set(std::move(timage));
    return std::move(file);
}

static auto generate_mipmaps_impl(Texture &&file) -> Result
{
    const auto &tex   = file.get();
    const auto &info  = tex.GetMetadata();
    const size_t mips = optimal_mip_count({.w = info.width, .h = info.height});

    ScratchImage timage;
    const auto hr = GenerateMipMaps(tex.GetImages(),
                                    tex.GetImageCount(),
                                    tex.GetMetadata(),
                                    DirectX::TEX_FILTER_SEPARATE_ALPHA,
                                    mips,
                                    timage);

    if (FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    file.set(std::move(timage));
    return std::move(file);
}

auto generate_mipmaps(Texture &&file) -> Result
{
    return prepare_generate_mipmaps(std::move(file)).and_then(generate_mipmaps_impl);
}

auto resize(Texture &&file, Dimension dim) -> Result
{
    const auto &tex  = file.get();
    const auto &info = tex.GetMetadata();

    ScratchImage timage;

    // DirectX::Resize is dumb. If WIC is used, it will convert the image to
    // R32G32B32A32 It works for small images... But will, for example, allocate
    // 1gb for a 8k picture. So disable WIC
    constexpr auto filter = DirectX::TEX_FILTER_SEPARATE_ALPHA | DirectX::TEX_FILTER_FORCE_NON_WIC;
    const auto hr         = Resize(tex.GetImages(), tex.GetImageCount(), info, dim.w, dim.h, filter, timage);
    if (FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    file.set(std::move(timage));
    return std::move(file);
}

auto bc1_counterpart(const DXGI_FORMAT bc3) noexcept -> DXGI_FORMAT
{
    switch (bc3)
    {
        case DXGI_FORMAT_BC3_UNORM: return DXGI_FORMAT_BC1_UNORM;
        case DXGI_FORMAT_BC3_UNORM_SRGB: return DXGI_FORMAT_BC1_UNORM_SRGB;
        case DXGI_FORMAT_BC3_TYPELESS: return DXGI_FORMAT_BC1_TYPELESS;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

namespace {
constexpr size_t k_bc3_block_bytes = 16;
constexpr size_t k_bc1_block_bytes = 8;

[[nodiscard]] auto block_count(const size_t pixels) noexcept -> size_t
{
    return std::max<size_t>(1, (pixels + 3) / 4);
}

/// One BC3 alpha block (8 bytes): every one of its 16 texels decodes to exactly 255.
[[nodiscard]] auto alpha_block_opaque(const uint8_t *block) noexcept -> bool
{
    const uint8_t a0 = block[0];
    const uint8_t a1 = block[1];

    uint64_t bits = 0;
    for (int i = 5; i >= 0; --i)
        bits = (bits << 8) | block[2 + i];

    for (int t = 0; t < 16; ++t)
    {
        const auto idx = static_cast<unsigned>((bits >> (3 * t)) & 7U);
        bool opaque    = false;
        if (idx == 0)
            opaque = a0 == 255;
        else if (idx == 1)
            opaque = a1 == 255;
        else if (a0 <= a1 && idx == 6)
            opaque = false; // explicit 0
        else if (a0 <= a1 && idx == 7)
            opaque = true; // explicit 255
        else
            opaque = a0 == 255 && a1 == 255; // interpolated: exact only when both ends are 255
        if (!opaque)
            return false;
    }
    return true;
}

/// One BC3 alpha block: every texel inside the `valid_w` x `valid_h` corner of the 4x4 block decodes to
/// exactly 0 or 255. Texels outside the image (block padding) are ignored. `has_zero` is set when a
/// checked texel is exactly 0.
[[nodiscard]] auto alpha_block_binary(const uint8_t *block,
                                      const size_t valid_w,
                                      const size_t valid_h,
                                      bool &has_zero) noexcept -> bool
{
    const uint8_t a0 = block[0];
    const uint8_t a1 = block[1];

    uint64_t bits = 0;
    for (int i = 5; i >= 0; --i)
        bits = (bits << 8) | block[2 + i];

    for (size_t t = 0; t < 16; ++t)
    {
        if (t % 4 >= valid_w || t / 4 >= valid_h)
            continue;
        const auto idx = static_cast<unsigned>((bits >> (3 * t)) & 7U);
        int value      = -1; // -1: not exactly 0 or 255
        if (idx == 0)
            value = a0;
        else if (idx == 1)
            value = a1;
        else if (a0 <= a1 && idx == 6)
            value = 0;
        else if (a0 <= a1 && idx == 7)
            value = 255;
        else if (a0 == a1)
            value = a0; // 6-alpha mode, equal ends: every interpolated entry equals the end
        if (value != 0 && value != 255)
            return false;
        if (value == 0)
            has_zero = true;
    }
    return true;
}

/// Colour half of a BC3 block (8 bytes at `c`) -> the equivalent BC1 block at `d`, bit-exact colours.
/// A BC3 block is decoded in 4-colour mode always; BC1 uses 4-colour mode only when c0 > c1.
void copy_colour_block(const uint8_t *c, uint8_t *d) noexcept
{
    const auto c0 = static_cast<uint16_t>(c[0] | (c[1] << 8));
    const auto c1 = static_cast<uint16_t>(c[2] | (c[3] << 8));
    uint32_t idx  = static_cast<uint32_t>(c[4]) | (static_cast<uint32_t>(c[5]) << 8)
                   | (static_cast<uint32_t>(c[6]) << 16) | (static_cast<uint32_t>(c[7]) << 24);

    uint16_t o0 = c0;
    uint16_t o1 = c1;
    if (c0 == c1)
    {
        idx = 0; // every 4-colour palette entry equals c0; index 0 avoids BC1's transparent slot
    }
    else if (c0 < c1)
    {
        // Swap endpoints: palette entries 0<->1 and 2<->3 trade places, so flip the low bit
        // of every 2-bit index. (2*a+b)/3 with the ends swapped is the same colour.
        o0 = c1;
        o1 = c0;
        idx ^= 0x55555555U;
    }

    d[0] = static_cast<uint8_t>(o0 & 0xFF);
    d[1] = static_cast<uint8_t>(o0 >> 8);
    d[2] = static_cast<uint8_t>(o1 & 0xFF);
    d[3] = static_cast<uint8_t>(o1 >> 8);
    d[4] = static_cast<uint8_t>(idx & 0xFF);
    d[5] = static_cast<uint8_t>((idx >> 8) & 0xFF);
    d[6] = static_cast<uint8_t>((idx >> 16) & 0xFF);
    d[7] = static_cast<uint8_t>(idx >> 24);
}
} // namespace

auto bc3_alpha_binary(const ScratchImage &tex) noexcept -> bool
{
    if (bc1_counterpart(tex.GetMetadata().format) == DXGI_FORMAT_UNKNOWN)
        return false;

    bool has_zero      = false;
    const auto *images = tex.GetImages();
    for (size_t i = 0; i < tex.GetImageCount(); ++i)
    {
        const auto &img = images[i];
        const auto bw   = block_count(img.width);
        const auto bh   = block_count(img.height);
        if (img.pixels == nullptr || img.rowPitch < bw * k_bc3_block_bytes)
            return false;
        for (size_t by = 0; by < bh; ++by)
        {
            const auto *row     = img.pixels + by * img.rowPitch;
            const auto valid_h  = std::min<size_t>(4, img.height - by * 4);
            for (size_t bx = 0; bx < bw; ++bx)
            {
                const auto valid_w = std::min<size_t>(4, img.width - bx * 4);
                if (!alpha_block_binary(row + bx * k_bc3_block_bytes, valid_w, valid_h, has_zero))
                    return false;
            }
        }
    }
    return has_zero;
}

auto is_diffuse_path(const Path &path) -> bool
{
    auto stem = path.stem().u8string();
    for (auto &ch : stem)
        if (ch >= u8'A' && ch <= u8'Z')
            ch = static_cast<char8_t>(ch - u8'A' + u8'a');

    constexpr std::u8string_view suffixes[] = {u8"_n",  u8"_g",  u8"_m",   u8"_e",  u8"_p",   u8"_s", u8"_sk",
                                               u8"_hh", u8"_hl", u8"_em",  u8"_msn", u8"_b",  u8"_bl"};
    const std::u8string_view s              = stem;
    for (const auto suffix : suffixes)
        if (s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix)
            return false;
    return true;
}

auto bc3_alpha_all_opaque(const ScratchImage &tex) noexcept -> bool
{
    if (bc1_counterpart(tex.GetMetadata().format) == DXGI_FORMAT_UNKNOWN)
        return false;

    const auto *images = tex.GetImages();
    for (size_t i = 0; i < tex.GetImageCount(); ++i)
    {
        const auto &img   = images[i];
        const auto bw     = block_count(img.width);
        const auto bh     = block_count(img.height);
        if (img.pixels == nullptr || img.rowPitch < bw * k_bc3_block_bytes)
            return false;
        for (size_t by = 0; by < bh; ++by)
        {
            const auto *row = img.pixels + by * img.rowPitch;
            for (size_t bx = 0; bx < bw; ++bx)
                if (!alpha_block_opaque(row + bx * k_bc3_block_bytes))
                    return false;
        }
    }
    return true;
}

auto bc3_to_bc1(Texture &&file) -> Result
{
    const auto &tex  = file.get();
    const auto &info = tex.GetMetadata();
    const auto out_format = bc1_counterpart(info.format);
    if (out_format == DXGI_FORMAT_UNKNOWN)
        return tl::make_unexpected(Error(TextureErr::BadInput));

    auto out_info   = info;
    out_info.format = out_format;
    // Alpha is gone; tell readers so (DDS header alpha mode).
    out_info.SetAlphaMode(DirectX::TEX_ALPHA_MODE_OPAQUE);

    ScratchImage out;
    if (const auto hr = out.Initialize(out_info); FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));
    if (out.GetImageCount() != tex.GetImageCount())
        return tl::make_unexpected(Error(TextureErr::Unknown));

    const auto *src_images = tex.GetImages();
    const auto *dst_images = out.GetImages();
    for (size_t i = 0; i < tex.GetImageCount(); ++i)
    {
        const auto &src = src_images[i];
        const auto &dst = dst_images[i];
        const auto bw   = block_count(src.width);
        const auto bh   = block_count(src.height);
        if (src.pixels == nullptr || dst.pixels == nullptr || src.width != dst.width
            || src.height != dst.height || src.rowPitch < bw * k_bc3_block_bytes
            || dst.rowPitch < bw * k_bc1_block_bytes)
            return tl::make_unexpected(Error(TextureErr::Unknown));

        for (size_t by = 0; by < bh; ++by)
        {
            const auto *src_row = src.pixels + by * src.rowPitch;
            auto *dst_row       = dst.pixels + by * dst.rowPitch;
            for (size_t bx = 0; bx < bw; ++bx)
            {
                copy_colour_block(src_row + bx * k_bc3_block_bytes + 8, dst_row + bx * k_bc1_block_bytes);
            }
        }
    }

    file.set(std::move(out));
    return std::move(file);
}

auto bc3_to_bc1a(Texture &&file) -> Result
{
    const auto &tex  = file.get();
    const auto &info = tex.GetMetadata();
    const auto out_format = bc1_counterpart(info.format);
    if (out_format == DXGI_FORMAT_UNKNOWN || info.format == DXGI_FORMAT_BC3_TYPELESS)
        return tl::make_unexpected(Error(TextureErr::BadInput));

    // Decode, then let DirectXTex encode to BC1 (alpha threshold 0.5: exact 0/255 maps to the
    // transparent / opaque slot with no ambiguity).
    ScratchImage rgba;
    if (const auto hr = Decompress(tex.GetImages(), tex.GetImageCount(), info, DXGI_FORMAT_UNKNOWN, rgba);
        FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));

    ScratchImage encoded;
    if (const auto hr = Compress(rgba.GetImages(),
                                 rgba.GetImageCount(),
                                 rgba.GetMetadata(),
                                 out_format,
                                 DirectX::TEX_COMPRESS_DEFAULT,
                                 0.5F,
                                 encoded);
        FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));
    if (encoded.GetImageCount() != tex.GetImageCount())
        return tl::make_unexpected(Error(TextureErr::Unknown));

    // Same layout, but with an alpha mode that doesn't claim the texture is opaque.
    auto out_info = encoded.GetMetadata();
    if (info.GetAlphaMode() == DirectX::TEX_ALPHA_MODE_OPAQUE)
        out_info.SetAlphaMode(DirectX::TEX_ALPHA_MODE_UNKNOWN);
    else
        out_info.SetAlphaMode(info.GetAlphaMode());

    ScratchImage out;
    if (const auto hr = out.Initialize(out_info); FAILED(hr))
        return tl::make_unexpected(error_from_hresult(hr));
    if (out.GetImageCount() != tex.GetImageCount())
        return tl::make_unexpected(Error(TextureErr::Unknown));

    const auto *src_images = tex.GetImages();
    const auto *enc_images = encoded.GetImages();
    const auto *dst_images = out.GetImages();
    for (size_t i = 0; i < tex.GetImageCount(); ++i)
    {
        const auto &src = src_images[i];
        const auto &enc = enc_images[i];
        const auto &dst = dst_images[i];
        const auto bw   = block_count(src.width);
        const auto bh   = block_count(src.height);
        if (src.pixels == nullptr || dst.pixels == nullptr || enc.pixels == nullptr || src.width != dst.width
            || src.height != dst.height || enc.width != dst.width || enc.height != dst.height
            || src.rowPitch < bw * k_bc3_block_bytes || dst.rowPitch < bw * k_bc1_block_bytes
            || enc.rowPitch < bw * k_bc1_block_bytes)
            return tl::make_unexpected(Error(TextureErr::Unknown));

        for (size_t by = 0; by < bh; ++by)
        {
            const auto *src_row = src.pixels + by * src.rowPitch;
            const auto *enc_row = enc.pixels + by * enc.rowPitch;
            auto *dst_row       = dst.pixels + by * dst.rowPitch;
            for (size_t bx = 0; bx < bw; ++bx)
            {
                const auto *s = src_row + bx * k_bc3_block_bytes;
                auto *d       = dst_row + bx * k_bc1_block_bytes;
                if (alpha_block_opaque(s))
                    copy_colour_block(s + 8, d); // fully opaque block: lossless
                else
                    std::memcpy(d, enc_row + bx * k_bc1_block_bytes, k_bc1_block_bytes);
            }
        }
    }

    file.set(std::move(out));
    return std::move(file);
}
} // namespace btu::tex

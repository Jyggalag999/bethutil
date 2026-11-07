/* Copyright (C) 2026 G'k
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
//
// Thin adapter between bethutil's span-in/vector-out convention and the shared
// kfcompress core (detail/kfcompress_core.hpp, src/kf/kfcompress_core.cpp), which
// is a verbatim copy of the standalone kfcompress CLI's own core. Keep those two
// files in sync with the kfcompress repo rather than editing them here.

#include "btu/kf/compress.hpp"

#include "btu/kf/detail/byteio.hpp"
#include "btu/kf/detail/kfcompress_core.hpp"
#include "btu/kf/error_code.hpp"

#include <btu/common/string.hpp>

#include <cstring>
#include <new>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace btu::kf {

auto compress(std::span<const std::byte> input,
              const Settings &settings,
              std::string *diagnostics,
              bool *modified) noexcept -> tl::expected<std::vector<std::byte>, Error>
{
    // This function is noexcept, and constructing an ostringstream (or copying its text out) can throw
    // bad_alloc, so none of that may sit outside a try. Hence the outer try: if even the log can't be
    // built or reported, all that is left to return is a bare Unknown.
    try
    {
        std::ostringstream log;
        const auto finish = [&] {
            if (diagnostics)
                *diagnostics = log.str();
        };

        try
        {
            kfc::Options opt;
            opt.ratio               = settings.ratio;
            opt.maxRotDeg           = settings.max_rot_deg;
            opt.maxTrans            = settings.max_trans;
            opt.maxScale            = settings.max_scale;
            opt.useTolerance        = settings.use_tolerance;
            opt.tolerance           = settings.tolerance;
            opt.skip                = settings.skip;
            opt.compact16           = settings.compact16;
            opt.forcedControlPoints = settings.control_points;
            opt.maxControlPoints    = settings.max_control_points;
            opt.minControlPoints    = settings.min_control_points;
            opt.noBlacklist         = settings.no_blacklist;
            opt.forceConvertAll     = settings.force_convert_all;
            opt.dedupe              = settings.dedupe;
            opt.dedupeEpsilon       = settings.dedupe_epsilon;
            for (const auto &name : settings.blacklist_add)
                opt.extraBlacklist.push_back(common::as_ascii_string(name));

            std::vector<uint8_t> bytes(input.size());
            if (!input.empty())
                std::memcpy(bytes.data(), input.data(), input.size());

            auto result = kfc::compressBytes(std::move(bytes), opt, log);

            std::vector<std::byte> out(result.bytes.size());
            if (!out.empty())
                std::memcpy(out.data(), result.bytes.data(), out.size());
            finish();
            if (modified)
                *modified = result.modified;
            return out;
        }
        // The core throws std::runtime_error with a specific reason (wrong NIF version, malformed
        // block table, ...). A std::error_code can't carry it, so hand it to the caller through
        // `diagnostics`. Only FormatError means "this isn't a file we understand"; anything else
        // (an internal invariant, length_error, ...) is a failed conversion of a valid-looking file,
        // and bad_alloc is a resource problem rather than anything about the file.
        catch (const FormatError &e)
        {
            log << "Error: " << e.what() << "\n";
            finish();
            return tl::make_unexpected(Error(KfErr::UnsupportedFormat));
        }
        catch (const std::bad_alloc &)
        {
            log << "Error: out of memory\n";
            finish();
            return tl::make_unexpected(Error(KfErr::Unknown));
        }
        catch (const std::exception &e)
        {
            log << "Error: " << e.what() << "\n";
            finish();
            return tl::make_unexpected(Error(KfErr::ConversionFailed));
        }
        catch (...)
        {
            log << "Error: unknown exception\n";
            finish();
            return tl::make_unexpected(Error(KfErr::Unknown));
        }
    }
    catch (...)
    {
        return tl::make_unexpected(Error(KfErr::Unknown));
    }
}

} // namespace btu::kf

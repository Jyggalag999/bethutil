/* Copyright (C) 2026 G'k
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
#pragma once

#include <btu/common/error.hpp>
#include <btu/common/json.hpp>
#include <tl/expected.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace btu::kf {
using common::Error;

/// @brief Settings for compressing a Gamebryo/NetImmerse .kf animation's
/// NiTransformInterpolator+NiTransformData channels into
/// NiBSplineTransformInterpolator/NiBSplineCompTransformInterpolator +
/// NiBSplineData/NiBSplineBasisData - the same B-spline curve compression
/// format Fallout 3/New Vegas's own engine already supports (this is what
/// F:\Kompresor CLI's kfcompress reconstructs, IDA/SDK/real-file-verified;
/// see its PROJECT_SUMMARY.md). Mirrors kfcompress's own --ratio/--tolerance/
/// --skip/--compact16/--control-points/--max-control-points/--blacklist-add/
/// --no-blacklist/--force-convert-all flags one-for-one.
struct Settings
{
    // Control-point count = round(ratio * original key count), one-shot
    // least-squares fit, no search - the actual Gamebryo NiAnimationCompression
    // algorithm per the official SDK docs, and empirically the closest match
    // to real vanilla-compressed files (see kfcompress's README/PROJECT_SUMMARY).
    double ratio = 0.80;

    // Switches to the old adaptive-search mode instead of ratio: find the
    // smallest control-point count whose max reconstruction error is <= tolerance.
    bool use_tolerance = false;
    double tolerance = 0.03;

    // Force a fixed control-point count per bone, overriding both ratio and
    // use_tolerance. 0 = auto (ratio- or tolerance-driven).
    int control_points = 0;

    // Hard cap on the adaptive search regardless of keyframe count - protects
    // against one unusually long/dense animation making a single channel's
    // O(M^3) fit take an unbounded amount of time. 0 disables the cap.
    int max_control_points = 400;

    // Keep every Nth source keyframe before fitting (1 = keep all).
    int skip = 1;

    // Pack control points as 16-bit fixed point (NiBSplineCompTransformInterpolator)
    // instead of raw floats.
    bool compact16 = false;

    // Convert every eligible bone even if doing so wouldn't actually shrink
    // that bone's data at the file's shared control-point count. Off by
    // default, matching observed vanilla behavior (see kfcompress's README).
    bool force_convert_all = false;

    // Convert every eligible bone, including ones normally excluded by the
    // built-in weapon/attach/IK/camera/shell-casing blacklist (see kfcompress's
    // isBlacklistedName). Off by default for a reason.
    bool no_blacklist = false;

    // Additional exact bone names to exclude, on top of the built-in blacklist.
    std::vector<std::u8string> blacklist_add;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Settings,
                                   ratio,
                                   use_tolerance,
                                   tolerance,
                                   control_points,
                                   max_control_points,
                                   skip,
                                   compact16,
                                   force_convert_all,
                                   no_blacklist,
                                   blacklist_add)

/// @brief Compresses one in-memory .kf/.nif file's eligible animation channels
/// per `settings`. Files with nothing eligible to convert (already B-spline,
/// no NiControllerSequence, etc.) come back byte-identical to `input`, same as
/// kfcompress's own "copy through unchanged" behavior - this always returns a
/// valid file, never "nothing to do", so callers don't need a separate
/// no-op case (compare btu::hkx::AnimExe::convert, which this replaces).
[[nodiscard]] auto compress(std::span<const std::byte> input, const Settings &settings) noexcept
    -> tl::expected<std::vector<std::byte>, Error>;

} // namespace btu::kf

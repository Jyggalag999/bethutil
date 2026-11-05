// Copyright (C) 2026 Jyggalag
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2 only,
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// SPDX-License-Identifier: GPL-2.0-only
// See LICENSE.txt for the full license text.
//
// kfcompress core: the whole per-file pipeline as bytes in -> bytes out, with
// no file or console I/O, so the CLI (main.cpp) and other front-ends (the
// Cathedral Assets Optimizer integration) share one implementation.

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace kfc {

struct Options {
    // ---- control-point count selection ----
    // Default mode is RATIO, matching the actual Gamebryo NiAnimationCompression
    // algorithm, not a guess: the official SDK docs (Gamebryo Artist's Guide,
    // 3ds Max Plug-in, "Animation Compression") state control-point count is
    // "what percentage of the original keys to use when fitting to a spline"
    // -- NiBSplineFit's constructor takes a fixed iOutQuantity and does a
    // one-shot least-squares fit to it, no adaptive tolerance search at all.
    // 0.80 as the default was confirmed empirically, not just from the docs:
    // measuring (control points / original key count) on the "driving" bone
    // of every one of 553 real vanilla-compressed files gives an exact,
    // sharp spike at ratio == 0.800 in 40% of files (217/536), with
    // secondary common settings around 0.50 and 0.425 matching the docs'
    // description of per-sequence/per-object ratio overrides.
    // --tolerance switches to the OLD adaptive-search mode instead (kept
    // for cases where matching vanilla's exact behavior matters less than
    // hitting a specific accuracy bound) -- see useTolerance below.
    //
    // [beta] What --ratio is actually multiplied by changed: analysis of
    // real vanilla-compressed files' actual structure showed the format
    // does NOT compute M as ratio * original_key_count. It resamples the source
    // curve at a fixed ~61Hz grid (duration-driven, not key-count-driven)
    // and computes M = round(0.5 * ratio * resampled_sample_count).
    // The two formulas are mathematically equivalent for ~30fps-authored
    // content (the vast majority of the corpus, and why the 0.800 spike
    // below still holds), but the old key-count-driven formula starved
    // sparse/sharp channels (few keys over a real duration) of control
    // points -- see fitToPolicy for the actual formula now in use.
    double ratio = 0.80;
    bool useTolerance = false;      // true once --tolerance is passed explicitly
    double tolerance = 0.03;        // only used when useTolerance is true
    int skip = 1;                   // keep every Nth source keyframe (1 = no skip)
    bool compact16 = false;
    int forcedControlPoints = 0;    // 0 = auto (ratio- or tolerance-driven)
    int maxControlPoints = 400;     // hard cap on adaptive search, regardless of
                                     // sample count -- protects against one
                                     // pathological (very long/dense) animation
                                     // making a single channel's O(M^3) fit take
                                     // an unbounded amount of time. 0 = no cap.
    int minControlPoints = 48;      // floor under --ratio's M, regardless of how
                                     // few samples that ratio implies -- found via
                                     // a real animation (EM_SprintF's sprint-forward
                                     // gait, ~33 source samples on Bip01 L Toe0)
                                     // where ratio=0.80 gave M=26 and the toe's
                                     // sharp push-off motion came back with 0.196
                                     // units of translation error, visibly wrong
                                     // in-game -- a short/snappy clip has too few
                                     // samples for 80% of them to be enough basis
                                     // functions to represent a sharp corner, even
                                     // though the same ratio is fine on longer,
                                     // smoother clips. 48 was verified against that
                                     // same file: it's above the bone's own ~33-
                                     // sample ceiling (so maxM clamps it down to
                                     // "just use every sample" for that bone), which
                                     // took the error from 0.196 to 1.1e-06 -- 32
                                     // only got partway (to 0.035). Only lowers
                                     // compression on short animations (never raises
                                     // M above the sample count itself or above
                                     // --max-control-points). 0 = no floor.
    bool verbose = false;
    bool dryRun = false;
    bool noBlacklist = false;
    std::vector<std::string> extraBlacklist;
    bool forceConvertAll = false; // skip the "would this actually shrink?" eligibility check
    bool dedupe = false; // run kfdedupe's lossless repeated-key removal as a pre-pass
    float dedupeEpsilon = 0.0f; // only meaningful with --dedupe; see --epsilon help text
    bool dedupeOnly = false; // run ONLY the dedupe pass, skip B-spline conversion entirely
                              // (this is what used to require the separate kfdedupe.exe;
                              // implies --dedupe). No bone blacklist applies here, and no
                              // NIF header/block parsing happens at all -- same scope and
                              // behavior as standalone kfdedupe, just reached via kfcompress.
};

struct Result {
    size_t bonesConverted = 0;
    size_t bonesSkippedBlacklist = 0;
    size_t bonesSkippedNotWorthwhile = 0;
    size_t dedupeBlocksTouched = 0;
    size_t dedupeKeysRemoved = 0;
    size_t inputBytes = 0, outputBytes = 0;
    bool wroteFile = false;  // false only for dry runs: `bytes` is then empty
    bool modified = false;   // `bytes` differs from the input (dedupe and/or B-spline conversion)
    std::vector<uint8_t> bytes; // the output file contents (== the input when !modified)
};

// Compresses one .kf file held in memory. Progress/diagnostics go to `log`.
// Throws std::runtime_error on unrecoverable errors (bad format, etc.); the
// message says why. Thread-safe: no shared state between calls.
Result compressBytes(const std::vector<uint8_t>& input, const Options& opt, std::ostream& log);

} // namespace kfc

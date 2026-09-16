/* Copyright (C) 2026 G'k
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
//
// This file is a near-verbatim port of F:\Kompresor CLI\src\main.cpp's
// convertFile() and everything it depends on (isBlacklistedName through
// appendChannel) - reverse-engineered and validated against 553 real
// vanilla-compressed files and a real 14,313-file mod batch (see that
// project's PROJECT_SUMMARY.md/README.md for the full account, including
// several real bugs found via in-game testing and fixed at the root cause).
// Deliberately NOT rewritten or "cleaned up" during this port: the goal is
// to carry that validated behavior over exactly, changing only how a single
// file's bytes get in and out (see compress() at the bottom, which adapts
// the original path-based Options/convertFile to bethutil's in-memory
// span-in/vector-out convention the same way btu::hkx::AnimExe::convert_impl
// already bridges a path-based external tool into that same convention).

#include "btu/kf/compress.hpp"

#include "btu/kf/detail/bspline.hpp"
#include "btu/kf/detail/niftypes.hpp"
#include "btu/kf/error_code.hpp"

#include <btu/common/filesystem.hpp>
#include <btu/common/string.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nif;
namespace fs = std::filesystem;

namespace btu::kf {
namespace {

// ---------------------------------------------------------------- name blacklist
//
// Mirrors the game/toolchain's own IsBlacklistedName() check: bones used for
// weapon attachment, IK end-effectors, or marked with "##" are excluded from
// lossy B-spline conversion by default, since those need exact (not
// approximated) transforms -- e.g. a weapon socket a few thousandths off
// is far more visible/gameplay-relevant than the same error on a spine bone.
bool isBlacklistedName(const std::string &name, const std::vector<std::string> &extra)
{
    // The first block (Weapon .. R Clavicle) is copied exactly from Wall's
    // own IsBlacklistedName (confirmed against his screenshots). Everything
    // after is this tool's own addition, not part of Wall's original list --
    // added after real in-game testing found visible breakage that traces
    // straight back to these gaps:
    //   - Any "camera" bone (substring match, not exact -- see below): never
    //     excluded, so it was getting lossy-compressed like anything else.
    //     Any error there is directly visible (it's what the player is
    //     looking through), unlike a limb bone where sub-millimeter error is
    //     lost in the noise -- there's no size/quality tradeoff where
    //     compressing this bone is worth it. Exhaustively scanning all
    //     14,313 real animation files for camera-ish names turned up not
    //     just "Camera1st"/"Camera3rd"/"NullCamera1st"/"NullCamera3rd" but
    //     also modder joke/variant names ("Camera1st gay", "notCamera3rd")
    //     that an exact-name list would keep missing one at a time forever
    //     -- a substring match on "camera" (case-insensitive) catches the
    //     whole family at once, present and future.
    //   - "Bip01 L Hand/Forearm/UpperArm/Clavicle": Wall's list only ever
    //     covers the right arm (the typical weapon-holding hand), but 1st-
    //     person reload animations routinely use the LEFT hand too (racking
    //     a slide, gripping a magazine/foregrip, holding the Pip-Boy on the
    //     wrist it's worn on) -- confirmed by real conversions showing up to
    //     0.019 rad rotation error on "Bip01 L Forearm" in a reload
    //     animation, at exactly the moment fingers need to align with a
    //     weapon part.
    static const char *kBuiltin[] = {
        "Weapon",
        "Bip01 WeaponBow",
        "B42AttachPointL",
        "ArrowNode",
        "Bip01 R Hand",
        "Bip01 R Forearm",
        "Bip01 R UpperArm",
        "Bip01 R Clavicle",
        "Bip01 L Hand",
        "Bip01 L Forearm",
        "Bip01 L UpperArm",
        "Bip01 L Clavicle",
    };
    for (auto *n : kBuiltin)
        if (name == n)
            return true;
    for (auto &n : extra)
        if (name == n)
            return true;
    if (name.find("##") != std::string::npos)
        return true;

    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (lower.find("camera") != std::string::npos)
        return true;

    // Shell casing / bullet / projectile prop bones: found by running our
    // OWN fidelity check (decode our compressed output, compare against
    // the source at every keyframe) across a real sample of the 14,313-file
    // batch -- these bones showed translation errors up to 6.76 engine
    // units (everything else in the same sample: ~0.0001-0.02), vastly
    // larger than ordinary lossy-compression residual. These are small,
    // fast-moving discrete props (a shell casing ejecting, a bullet
    // visible in a chamber) that the player directly tracks visually --
    // same category as camera/weapon bones, just missed until this fit-
    // quality check (structural validation alone can't catch this kind of
    // bug: the output is bounds-correct and NaN-free, just numerically
    // wrong). Substring match, not exact names, because real files use a
    // wide spread of variants and typos for the same concept: "Bullet",
    // "Bullet0/1", "ShellCasingNode", "DummyShell", "NoShellHere",
    // "riotshell_OLD", "ProjectileNode" (+ "_Fire"/"_HeadLaser"/
    // "_RGatling"/"_SonicBark" suffixes), even "ProjectileNoo" (a typo).
    for (const char *frag : {"shell", "bullet", "projectile", "casing", "round"})
    {
        if (lower.find(frag) != std::string::npos)
            return true;
    }
    return false;
}

// ---------------------------------------------------------------- options
//
// Trimmed from kfcompress's own Options to exactly the fields convertFile()
// itself reads - batch/threading/in-place/backup fields (inDir/outDir/
// recursive/inPlace/noBackup/forceOverwrite/threads) belong to kfcompress's
// own standalone CLI driver, not to a single-file conversion, and CAO's own
// Manager/ThreadPool already parallelizes per-file work at a higher level.
struct Options
{
    std::string inPath, outPath;
    double ratio             = 0.80;
    bool useTolerance        = false;
    double tolerance         = 0.03;
    int skip                 = 1;
    bool compact16           = false;
    int forcedControlPoints  = 0;
    int maxControlPoints     = 400;
    bool verbose             = false;
    bool dryRun              = false;
    bool noBlacklist         = false;
    std::vector<std::string> extraBlacklist;
    bool forceConvertAll = false;
};

// ---------------------------------------------------------------- quaternion helpers

struct Quat
{
    double w = 1, x = 0, y = 0, z = 0;
};

Quat eulerXYZToQuat(double rx, double ry, double rz)
{
    // Extrinsic X-then-Y-then-Z: q = qz * qy * qx
    double cx = std::cos(rx * 0.5), sx = std::sin(rx * 0.5);
    double cy = std::cos(ry * 0.5), sy = std::sin(ry * 0.5);
    double cz = std::cos(rz * 0.5), sz = std::sin(rz * 0.5);
    Quat qx{cx, sx, 0, 0}, qy{cy, 0, sy, 0}, qz{cz, 0, 0, sz};
    auto mul = [](const Quat &a, const Quat &b) {
        return Quat{a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
    };
    return mul(qz, mul(qy, qx));
}

// ---------------------------------------------------------------- sampling

struct Sample
{
    double t; // raw time, not yet normalized
    std::vector<double> v;
};

// Linear interpolation lookup within a sorted float KeyGroup (used to
// resample XYZ-euler axes onto a shared time base). Falls back to boundary
// clamping outside the key range.
double sampleFloatKeyGroup(const KeyGroup<float> &g, double t)
{
    if (g.keys.empty())
        return 0.0;
    if (g.keys.size() == 1 || t <= g.keys.front().time)
        return g.keys.front().value;
    if (t >= g.keys.back().time)
        return g.keys.back().value;
    for (size_t i = 0; i + 1 < g.keys.size(); ++i)
    {
        double t0 = g.keys[i].time, t1 = g.keys[i + 1].time;
        if (t >= t0 && t <= t1)
        {
            double a = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
            return g.keys[i].value * (1 - a) + g.keys[i + 1].value * a;
        }
    }
    return g.keys.back().value;
}

std::vector<double> decimateKeepEnds(const std::vector<double> &times, int skip)
{
    std::vector<double> keep;
    if (times.empty())
        return keep;
    for (size_t i = 0; i < times.size(); i += static_cast<size_t>(std::max(1, skip)))
        keep.push_back(times[i]);
    if (keep.back() != times.back())
        keep.push_back(times.back());
    return keep;
}

// ---------------------------------------------------------------- conversion job

struct ChannelFit
{
    bool present         = false;
    int numComponents    = 0;
    bspline::FitResult fit;
    float startTime = 0, stopTime = 1;
};

struct BoneJob
{
    int interpBlockIdx = -1;
    int dataBlockIdx   = -1;
    std::string name;
    TransformInterpolator interp;
    ChannelFit translation, rotation, scale;
};

// A handful of real mod files ship literal NaN/Inf keyframe values (found
// via a 14,313-file real batch: "Bip01 RPauldron" in one mod's idle
// animations has a NaN translation+rotation on its very first key --
// pre-existing garbage in that file, not something introduced here). NaN
// silently poisons every downstream computation (the least-squares fit,
// the min/max range used for --compact16's offset/halfRange), eventually
// surfacing as an infinite halfRange far from where the actual bad value
// was -- so check at the source instead of trying to catch it later.
bool allFinite(const std::vector<std::vector<double>> &vals)
{
    for (auto &row : vals)
        for (double v : row)
            if (!std::isfinite(v))
                return false;
    return true;
}

// Computes the fit for a channel's already-resampled (ts, vals) pair,
// choosing M by whichever mode is active: an explicit forcedM (highest
// priority, e.g. --control-points), a fixed ratio of the sample count
// (default -- matches NiBSplineFit's real fixed-iOutQuantity, one-shot
// least-squares fit, no search), or the old tolerance-driven adaptive
// search (opt-in via --tolerance).
bspline::FitResult fitToPolicy(const std::vector<double> &ts,
                               const std::vector<std::vector<double>> &vals,
                               int forcedM,
                               double ratio,
                               bool useTolerance,
                               double tolerance,
                               int maxControlPoints)
{
    int maxM = static_cast<int>(ts.size());
    if (maxControlPoints > 0)
        maxM = std::min(maxM, maxControlPoints);
    int minM = bspline::kDegree + 1;

    if (forcedM > 0)
        return bspline::fitBSpline(ts, vals, forcedM);
    if (useTolerance)
        return bspline::fitBSplineAdaptive(ts, vals, tolerance, minM, maxM);

    int m = static_cast<int>(std::lround(ratio * static_cast<double>(ts.size())));
    m     = std::clamp(m, minM, maxM);
    return bspline::fitBSpline(ts, vals, m);
}

// Computes the bone-wide parametric time range across ALL of a bone's
// present channels (translation/rotation/scale) BEFORE any fitting.
// NiBSplineCompTransformInterpolator has exactly ONE start/stop time per
// bone -- one shared parametric axis that every channel's control points
// are evaluated against at playback (see NiBSplineInterpolator's decode
// path) -- so every channel MUST be fit with its samples normalized into
// that same shared [0,1] range. Previously each channel normalized using
// its OWN first/last key time, which is only correct when every channel
// on a bone happens to start and stop at exactly the same time; any
// mismatch (extremely common -- e.g. a scale track with a single static
// key somewhere in the middle of the clip, or a rotation track that
// starts later than translation) silently fit that channel's control
// points against the WRONG parametric axis. The output was structurally
// valid and passed every bounds/NaN check, but decoded numerically wrong
// for most of the clip -- worst case a single-key channel gets its lone
// sample normalized to t=0 regardless of where it actually falls, so
// nearly all of its shared control points solve to 0 via the Tikhonov
// regularization and the channel decodes near 0 (e.g. scale collapsing
// toward 0, i.e. a "smurf-size" character) everywhere except right at the
// start of the animation.
void computeBoneTimeRange(const TransformData &d, double &outT0, double &outT1)
{
    double t0 = 1e30, t1 = -1e30;
    auto extend = [&](double t) {
        t0 = std::min(t0, t);
        t1 = std::max(t1, t);
    };
    for (auto &k : d.translations.keys) extend(k.time);
    if (d.hasRotation())
    {
        if (!d.isXYZRotation())
        {
            for (auto &k : d.quatKeys) extend(k.time);
        }
        else
        {
            for (int a = 0; a < 3; ++a)
                for (auto &k : d.xyzRotations[a].keys) extend(k.time);
        }
    }
    for (auto &k : d.scales.keys) extend(k.time);
    if (t0 > t1)
    {
        t0 = 0;
        t1 = 1;
    }
    outT0 = t0;
    outT1 = t1;
}

ChannelFit fitVectorChannel(const KeyGroup<Vector3> &g,
                            int skip,
                            double ratio,
                            bool useTolerance,
                            double tolerance,
                            int forcedM,
                            int maxControlPoints,
                            double sharedT0,
                            double sharedT1)
{
    ChannelFit cf;
    if (!g.present())
        return cf;
    std::vector<double> times;
    for (auto &k : g.keys) times.push_back(k.time);
    auto keptTimes = decimateKeepEnds(times, skip);

    double t0 = sharedT0, t1 = sharedT1;
    std::vector<double> ts;
    std::vector<std::vector<double>> vals;
    for (double t : keptTimes)
    {
        // find matching original key (decimation keeps exact original samples)
        auto it = std::lower_bound(g.keys.begin(), g.keys.end(), t, [](const Key<Vector3> &k, double tt) {
            return k.time < tt;
        });
        Vector3 v  = (it != g.keys.end()) ? it->value : g.keys.back().value;
        double tn  = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
        ts.push_back(tn);
        vals.push_back({v.x, v.y, v.z});
    }
    if (!allFinite(vals))
        return ChannelFit{}; // NaN/Inf in source keys -- leave uncompressed rather than propagate garbage
    cf.fit            = fitToPolicy(ts, vals, forcedM, ratio, useTolerance, tolerance, maxControlPoints);
    cf.present        = true;
    cf.numComponents  = 3;
    cf.startTime      = static_cast<float>(t0);
    cf.stopTime       = static_cast<float>(t1);
    return cf;
}

ChannelFit fitScaleChannel(const KeyGroup<float> &g,
                           int skip,
                           double ratio,
                           bool useTolerance,
                           double tolerance,
                           int forcedM,
                           int maxControlPoints,
                           double sharedT0,
                           double sharedT1)
{
    ChannelFit cf;
    if (!g.present())
        return cf;
    std::vector<double> times;
    for (auto &k : g.keys) times.push_back(k.time);
    auto keptTimes = decimateKeepEnds(times, skip);
    double t0 = sharedT0, t1 = sharedT1;
    std::vector<double> ts;
    std::vector<std::vector<double>> vals;
    for (double t : keptTimes)
    {
        auto it = std::lower_bound(g.keys.begin(), g.keys.end(), t, [](const Key<float> &k, double tt) {
            return k.time < tt;
        });
        float v   = (it != g.keys.end()) ? it->value : g.keys.back().value;
        double tn = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
        ts.push_back(tn);
        vals.push_back({v});
    }
    if (!allFinite(vals))
        return ChannelFit{}; // NaN/Inf in source keys -- leave uncompressed rather than propagate garbage
    cf.fit           = fitToPolicy(ts, vals, forcedM, ratio, useTolerance, tolerance, maxControlPoints);
    cf.present       = true;
    cf.numComponents = 1;
    cf.startTime     = static_cast<float>(t0);
    cf.stopTime      = static_cast<float>(t1);
    return cf;
}

ChannelFit fitRotationChannel(const TransformData &d,
                              int skip,
                              double ratio,
                              bool useTolerance,
                              double tolerance,
                              int forcedM,
                              int maxControlPoints,
                              double sharedT0,
                              double sharedT1)
{
    ChannelFit cf;
    if (!d.hasRotation())
        return cf;

    std::vector<double> rawTimes;
    std::vector<Quat> rawQuats;

    if (!d.isXYZRotation())
    {
        for (auto &k : d.quatKeys)
        {
            rawTimes.push_back(k.time);
            rawQuats.push_back({k.value.w, k.value.x, k.value.y, k.value.z});
        }
    }
    else
    {
        std::set<float> timeSet;
        for (int a = 0; a < 3; ++a)
            for (auto &k : d.xyzRotations[a].keys) timeSet.insert(k.time);
        for (float t : timeSet)
        {
            double rx = sampleFloatKeyGroup(d.xyzRotations[0], t);
            double ry = sampleFloatKeyGroup(d.xyzRotations[1], t);
            double rz = sampleFloatKeyGroup(d.xyzRotations[2], t);
            Quat q    = eulerXYZToQuat(rx, ry, rz);
            rawTimes.push_back(t);
            rawQuats.push_back(q);
        }
    }
    if (rawTimes.empty())
        return cf;

    // Enforce quaternion sign continuity (shortest-path) before fitting so
    // the fit doesn't chase a sign flip as a huge jump.
    for (size_t i = 1; i < rawQuats.size(); ++i)
    {
        double dot = rawQuats[i - 1].w * rawQuats[i].w + rawQuats[i - 1].x * rawQuats[i].x
                    + rawQuats[i - 1].y * rawQuats[i].y + rawQuats[i - 1].z * rawQuats[i].z;
        if (dot < 0)
        {
            rawQuats[i].w *= -1;
            rawQuats[i].x *= -1;
            rawQuats[i].y *= -1;
            rawQuats[i].z *= -1;
        }
    }

    auto keptTimes = decimateKeepEnds(rawTimes, skip);
    double t0 = sharedT0, t1 = sharedT1;
    std::vector<double> ts;
    std::vector<std::vector<double>> vals;
    for (double t : keptTimes)
    {
        size_t idx = std::lower_bound(rawTimes.begin(), rawTimes.end(), t) - rawTimes.begin();
        if (idx >= rawQuats.size())
            idx = rawQuats.size() - 1;
        Quat q    = rawQuats[idx];
        double tn = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
        ts.push_back(tn);
        vals.push_back({q.w, q.x, q.y, q.z});
    }
    if (!allFinite(vals))
        return ChannelFit{}; // NaN/Inf in source keys -- leave uncompressed rather than propagate garbage
    cf.fit           = fitToPolicy(ts, vals, forcedM, ratio, useTolerance, tolerance, maxControlPoints);
    cf.present       = true;
    cf.numComponents = 4;
    cf.startTime     = static_cast<float>(t0);
    cf.stopTime      = static_cast<float>(t1);
    return cf;
}

// Appends one channel's control points into the shared pool (float or
// compact int16), returning the element offset ("Handle") of its first
// control point.
uint32_t appendChannel(const ChannelFit &cf,
                       bool compact,
                       std::vector<float> &floatPool,
                       std::vector<int16_t> &compactPool,
                       float &outOffset,
                       float &outHalfRange)
{
    outOffset = 0;
    outHalfRange = 1;
    if (!cf.present)
        return kNoHandle;

    if (!compact)
    {
        uint32_t handle = static_cast<uint32_t>(floatPool.size());
        for (auto &row : cf.fit.controlPoints)
            for (double v : row) floatPool.push_back(static_cast<float>(v));
        return handle;
    }

    double lo = 1e300, hi = -1e300;
    for (auto &row : cf.fit.controlPoints)
        for (double v : row)
        {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    double offset    = (lo + hi) * 0.5;
    double halfRange = std::max({std::fabs(hi - offset), std::fabs(lo - offset), 1e-6});
    outOffset         = static_cast<float>(offset);
    outHalfRange       = static_cast<float>(halfRange);

    uint32_t handle = static_cast<uint32_t>(compactPool.size());
    for (auto &row : cf.fit.controlPoints)
    {
        for (double v : row)
        {
            double n  = (v - offset) / halfRange; // [-1, 1]
            long q    = std::lround(n * 32767.0);
            q         = std::clamp<long>(q, -32767, 32767);
            compactPool.push_back(static_cast<int16_t>(q));
        }
    }
    return handle;
}

// ---------------------------------------------------------------- convertFile
//
// Does the actual work for one file. Writes progress/results into `log`
// rather than directly to cout/cerr so batch mode can run many of these
// concurrently (one per worker thread) and print each file's report as an
// atomic block afterwards, instead of interleaving output across threads.
// Throws std::runtime_error on unrecoverable errors (bad format, I/O
// failure, etc.) -- compress() below catches this and maps it to a KfErr.
struct ConvertResult
{
    size_t bonesConverted           = 0;
    size_t bonesSkippedBlacklist    = 0;
    size_t bonesSkippedNotWorthwhile = 0;
    size_t inputBytes = 0, outputBytes = 0;
    bool wroteFile = false;
};

ConvertResult convertFile(const Options &opt, std::ostream &log)
{
    ConvertResult result;

    std::ifstream f(opt.inPath, std::ios::binary);
    if (!f)
        throw std::runtime_error("Can't open input file: " + opt.inPath);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    result.inputBytes = raw.size();

    ByteReader hr(raw);
    Header header      = readHeader(hr);
    uint32_t bsVersion = header.bsHeader.bsVersion;

    // Every blockTypeIndex[i] is a raw uint16 read straight from file
    // bytes -- on a malformed, truncated, or non-matching file it can be
    // any value up to 65535, with nothing stopping it from pointing past
    // the end of blockTypes. Indexing a std::vector with operator[] does
    // NOT bounds-check, so an out-of-range value here is undefined
    // behavior -- a hard crash, not a catchable exception -- rather than
    // the clean error every other validation in this file produces.
    // Caught this the hard way running against a large (14k+ file) real
    // batch: one bad file took the whole process down with zero output,
    // because the crash happened inside a worker thread before any
    // per-file try/catch could see it.
    for (uint32_t i = 0; i < header.numBlocks; ++i)
    {
        if (header.blockTypeIndex[i] >= header.blockTypes.size())
        {
            throw std::runtime_error(
                "Block " + std::to_string(i) + " has an out-of-range block-type index "
                "(" + std::to_string(header.blockTypeIndex[i]) + " >= "
                + std::to_string(header.blockTypes.size())
                + " known types) -- file is "
                  "corrupt, truncated, or not actually this NIF version despite its extension.");
        }
    }

    // Compute per-block byte offsets within the file (blocks are
    // sequential right after the header, sized per header.blockSizes).
    std::vector<size_t> blockOffset(header.numBlocks);
    size_t cursor = hr.pos;
    for (uint32_t i = 0; i < header.numBlocks; ++i)
    {
        blockOffset[i] = cursor;
        cursor += header.blockSizes[i];
        // A block whose declared offset+size runs past the actual file
        // means every ByteReader constructed over that block's claimed
        // byte range would think it has more valid memory than it really
        // does (ByteReader::need() only checks against the *claimed*
        // size passed to it, not the real buffer bounds) -- a heap
        // buffer over-read. Must hard-fail here, not just warn, and
        // before any block gets read.
        if (cursor > raw.size())
        {
            throw std::runtime_error(
                "Block " + std::to_string(i) + "'s declared size runs past the end of "
                "the file (file is truncated or the block-size table is corrupt).");
        }
    }
    if (cursor != raw.size())
    {
        log << "Warning: " << (raw.size() - cursor)
            << " trailing bytes after the last declared block; keeping them as-is.\n";
    }
    size_t trailerStart = cursor, trailerLen = raw.size() - cursor;

    auto blockType = [&](int idx) -> const std::string & {
        return header.blockTypes[header.blockTypeIndex[idx]];
    };

    auto resolveString = [&](int32_t idx) -> std::string {
        if (idx < 0 || idx >= static_cast<int32_t>(header.strings.size()))
            return "";
        return header.strings[idx];
    };

    // ---- Find every NiControllerSequence, and every NiTransformInterpolator
    //      it references, and build one BoneJob per unique interpolator. ----
    std::vector<BoneJob> jobs;
    std::set<int> seenInterp;
    std::vector<int> seqIndices;
    size_t skippedByBlacklist = 0;
    for (uint32_t i = 0; i < header.numBlocks; ++i)
        if (blockType(i) == "NiControllerSequence")
            seqIndices.push_back(static_cast<int>(i));

    for (int seqIdx : seqIndices)
    {
        ByteReader sr(raw.data() + blockOffset[seqIdx], header.blockSizes[seqIdx]);
        ControllerSequence seq = ControllerSequence::read(sr, bsVersion, header.blockSizes[seqIdx]);
        for (auto &cb : seq.blocks)
        {
            int interpIdx = cb.interpolatorRef;
            if (interpIdx < 0 || interpIdx >= static_cast<int>(header.numBlocks))
                continue;
            if (blockType(interpIdx) != "NiTransformInterpolator")
                continue; // already bspline, or other type
            if (seenInterp.count(interpIdx))
                continue;
            seenInterp.insert(interpIdx);

            std::string nodeName = resolveString(cb.nodeNameIdx);
            if (!opt.noBlacklist && isBlacklistedName(nodeName, opt.extraBlacklist))
            {
                skippedByBlacklist++;
                if (opt.verbose)
                    log << "Skipping bone \"" << nodeName << "\" (interp#" << interpIdx
                        << "): blacklisted name\n";
                continue;
            }

            ByteReader ir(raw.data() + blockOffset[interpIdx], header.blockSizes[interpIdx]);
            TransformInterpolator ti = TransformInterpolator::read(ir, header.blockSizes[interpIdx]);
            int dataIdx              = ti.dataRef;
            if (dataIdx < 0 || dataIdx >= static_cast<int>(header.numBlocks))
                continue;
            if (blockType(dataIdx) != "NiTransformData" && blockType(dataIdx) != "NiKeyframeData")
                continue;

            BoneJob job;
            job.interpBlockIdx = interpIdx;
            job.dataBlockIdx   = dataIdx;
            job.interp         = ti;
            job.name           = nodeName;
            jobs.push_back(job);
        }
    }
    result.bonesSkippedBlacklist = skippedByBlacklist;
    if (skippedByBlacklist && !opt.verbose)
    {
        log << "Skipped " << skippedByBlacklist
            << " blacklisted bone(s) "
               "(weapon/attach/IK names, or containing \"##\"); use -v to see which, "
               "or --no-blacklist to convert them anyway.\n";
    }

    if (jobs.empty())
    {
        log << "No convertible NiTransformInterpolator/NiTransformData "
               "channels found (file may already use B-splines, or "
               "contains no NiControllerSequence).\n";
        if (!opt.dryRun)
        {
            // Copy the file through unchanged rather than writing nothing
            // at all -- in batch mode, every source file should have a
            // corresponding output file, even if there was nothing in it
            // to compress. Writing nothing here was a real gap: found
            // when a 14,313-file real batch produced only ~12,000 output
            // files, because every file with no convertible bones (some
            // camera/text-key-only sequences, files that already use
            // B-splines, etc.) simply vanished from the mirrored output
            // instead of being copied through.
            if (opt.inPath != opt.outPath)
            {
                std::ofstream of(opt.outPath, std::ios::binary);
                if (!of)
                    throw std::runtime_error("Can't open output file: " + opt.outPath);
                of.write(reinterpret_cast<const char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
                of.close();
                log << "Copied through unchanged: " << opt.outPath << "\n";
            }
            result.wroteFile   = true;
            result.outputBytes = raw.size();
        }
        return result;
    }

    // ---- Fit each job's channels ----
    // Pass 1: adaptive per-channel fit (finds each channel's own
    // minimal M under --tolerance). Real Bethesda-exported files share
    // ONE NiBSplineBasisData (and therefore one control-point count M)
    // across every channel of every bone in the sequence -- confirmed
    // by inspecting actual game .kf files -- so unless the caller forced
    // a fixed --control-points value (which already makes pass 1
    // uniform), pass 2 below re-fits every channel at the global max M
    // so they can all share a single basis block, matching that
    // behavior instead of inventing an untested per-bone variant.
    size_t origKeyBytes = 0;
    std::vector<TransformData> jobData(jobs.size());
    // Each bone's own control-point count M (NOT shared across the whole
    // file). The interpolator format has a single basisDataRef per bone --
    // translation/rotation/scale can't point at different basis blocks --
    // so every present channel of ONE bone must share the same M, but
    // different bones are free to use different M values, and do in real
    // vanilla files: measuring real compressed .kf files shows 53% of them
    // (294/553) contain more than one NiBSplineBasisData block, which the
    // previous "one M forced across the entire file" design couldn't ever
    // produce. Bones are grouped into basis blocks by matching M value
    // further down, rather than one shared block for everyone.
    std::vector<int> jobM(jobs.size(), 0);

    for (size_t j = 0; j < jobs.size(); ++j)
    {
        auto &job = jobs[j];
        ByteReader dr(raw.data() + blockOffset[job.dataBlockIdx], header.blockSizes[job.dataBlockIdx]);
        jobData[j] = TransformData::read(dr, header.blockSizes[job.dataBlockIdx]);
        origKeyBytes += header.blockSizes[job.dataBlockIdx];

        // Every channel on this bone must be normalized into the SAME
        // parametric [0,1] range before fitting -- see computeBoneTimeRange's
        // comment. Computed once, up front, and threaded through both
        // fitting passes below.
        double boneT0, boneT1;
        computeBoneTimeRange(jobData[j], boneT0, boneT1);

        // Pass 1: fit each present channel independently to find this
        // bone's own ideal per-channel M.
        job.translation = fitVectorChannel(jobData[j].translations,
                                           opt.skip,
                                           opt.ratio,
                                           opt.useTolerance,
                                           opt.tolerance,
                                           opt.forcedControlPoints,
                                           opt.maxControlPoints,
                                           boneT0,
                                           boneT1);
        job.rotation     = fitRotationChannel(jobData[j],
                                          opt.skip,
                                          opt.ratio,
                                          opt.useTolerance,
                                          opt.tolerance,
                                          opt.forcedControlPoints,
                                          opt.maxControlPoints,
                                          boneT0,
                                          boneT1);
        job.scale        = fitScaleChannel(jobData[j].scales,
                                    opt.skip,
                                    opt.ratio,
                                    opt.useTolerance,
                                    opt.tolerance,
                                    opt.forcedControlPoints,
                                    opt.maxControlPoints,
                                    boneT0,
                                    boneT1);

        int boneM = opt.forcedControlPoints > 0 ? opt.forcedControlPoints : bspline::kDegree + 1;
        if (job.translation.present)
            boneM = std::max(boneM, (int)job.translation.fit.controlPoints.size());
        if (job.rotation.present)
            boneM = std::max(boneM, (int)job.rotation.fit.controlPoints.size());
        if (job.scale.present)
            boneM = std::max(boneM, (int)job.scale.fit.controlPoints.size());
        jobM[j] = boneM;

        // Pass 2 (per-bone, not file-wide): re-fit any channel whose own M
        // came in below this bone's shared M.
        if (job.translation.present && (int)job.translation.fit.controlPoints.size() != boneM)
            job.translation = fitVectorChannel(jobData[j].translations,
                                               opt.skip,
                                               opt.ratio,
                                               opt.useTolerance,
                                               opt.tolerance,
                                               boneM,
                                               opt.maxControlPoints,
                                               boneT0,
                                               boneT1);
        if (job.rotation.present && (int)job.rotation.fit.controlPoints.size() != boneM)
            job.rotation = fitRotationChannel(
                jobData[j], opt.skip, opt.ratio, opt.useTolerance, opt.tolerance, boneM, opt.maxControlPoints, boneT0, boneT1);
        if (job.scale.present && (int)job.scale.fit.controlPoints.size() != boneM)
            job.scale = fitScaleChannel(jobData[j].scales,
                                        opt.skip,
                                        opt.ratio,
                                        opt.useTolerance,
                                        opt.tolerance,
                                        boneM,
                                        opt.maxControlPoints,
                                        boneT0,
                                        boneT1);

        // Every present channel MUST come out of pass 2 with exactly boneM
        // control points -- they all share one basis block (one M) per the
        // file format, so anything less means appendChannel will write
        // fewer elements into the pool than that basis block declares,
        // which is a silent out-of-bounds read at playback, not a
        // cosmetic inaccuracy. This exact scenario (a sparse channel, e.g.
        // a 2-key scale track, sharing a bone with a much denser one) is
        // what caused real, widespread in-game breakage in a 553- and then
        // 14,313-file real batch -- see bspline.hpp's fitBSpline comment
        // for the actual root cause (an internal M-vs-sample-count clamp).
        // Left as a hard failure rather than a silent clamp/skip: if this
        // ever fires again, it means that fix regressed or a new code path
        // bypassed fitToPolicy, and producing a corrupt file silently is
        // worse than aborting the whole conversion.
        if ((job.translation.present && (int)job.translation.fit.controlPoints.size() != boneM)
            || (job.rotation.present && (int)job.rotation.fit.controlPoints.size() != boneM)
            || (job.scale.present && (int)job.scale.fit.controlPoints.size() != boneM))
        {
            throw std::runtime_error(
                "Internal error: bone \"" + job.name + "\" (interp#" + std::to_string(job.interpBlockIdx)
                + ") has a channel whose fit didn't come back at the bone's shared M=" + std::to_string(boneM)
                + " -- refusing to write a file that would corrupt at playback. This means "
                  "bspline::fitBSpline's M-vs-sample-count contract regressed; see its comment.");
        }
    }

    if (opt.useTolerance && opt.maxControlPoints > 0)
    {
        bool anyOverTolerance = false;
        for (size_t j = 0; j < jobs.size(); ++j)
        {
            if (jobM[j] < opt.maxControlPoints)
                continue;
            auto &job = jobs[j];
            if (job.translation.present && job.translation.fit.maxError > opt.tolerance)
                anyOverTolerance = true;
            if (job.rotation.present && job.rotation.fit.maxError > opt.tolerance)
                anyOverTolerance = true;
            if (job.scale.present && job.scale.fit.maxError > opt.tolerance)
                anyOverTolerance = true;
        }
        if (anyOverTolerance)
        {
            log << "Warning: hit --max-control-points (" << opt.maxControlPoints
                << ") before every channel met --tolerance " << opt.tolerance
                << " on at least one bone -- some channels are less accurate "
                   "than requested. This usually means an unusually long/dense "
                   "source animation; raise --max-control-points if you need "
                   "tighter accuracy (fit cost grows with its cube, so raise "
                   "it cautiously).\n";
        }
    }

    // ---- Drop bones where B-spline conversion at the shared M wouldn't
    //      actually shrink the file. Real Bethesda-compressed files
    //      consistently leave low-motion bones (typically 2-key channels)
    //      as plain NiTransformInterpolator rather than pay control-point
    //      overhead for zero benefit. We don't have Wall's exact
    //      eligibility formula (it isn't in FalloutNV.exe), so this
    //      reconstructs its *effect* by computing the actual net file-size
    //      delta of converting this bone: converting replaces a 36-byte
    //      NiTransformInterpolator body with an 84-byte (60 if not
    //      --compact16) NiBSplineCompTransformInterpolator body, shrinks
    //      the original NiTransformData block to a 12-byte empty
    //      placeholder (see TransformData::writeEmpty), and adds this
    //      bone's control points to the shared pool.
    if (!opt.forceConvertAll)
    {
        size_t bytesPerComponent      = opt.compact16 ? 2 : 4;
        size_t newInterpBodySize      = opt.compact16 ? 84 : 60;
        constexpr size_t kOldInterpBodySize = 36; // NiQuatTransform(32) + dataRef(4)
        constexpr size_t kShrunkDataSize    = 12; // TransformData::writeEmpty()
        std::vector<BoneJob> keptJobs;
        std::vector<TransformData> keptJobData;
        std::vector<int> keptJobM;
        keptJobs.reserve(jobs.size());
        keptJobData.reserve(jobs.size());
        keptJobM.reserve(jobs.size());
        size_t skippedNotWorthwhile = 0;
        for (size_t j = 0; j < jobs.size(); ++j)
        {
            auto &job    = jobs[j];
            int m        = jobM[j];
            size_t poolBytes = 0;
            if (job.translation.present)
                poolBytes += static_cast<size_t>(m) * 3 * bytesPerComponent;
            if (job.rotation.present)
                poolBytes += static_cast<size_t>(m) * 4 * bytesPerComponent;
            if (job.scale.present)
                poolBytes += static_cast<size_t>(m) * 1 * bytesPerComponent;
            size_t originalDataBytes = header.blockSizes[job.dataBlockIdx];
            // Net change to the file if this bone IS converted (positive = grows).
            long long netDelta = static_cast<long long>(newInterpBodySize)
                                - static_cast<long long>(kOldInterpBodySize)
                                + static_cast<long long>(kShrunkDataSize)
                                - static_cast<long long>(originalDataBytes)
                                + static_cast<long long>(poolBytes);
            if (netDelta >= 0)
            {
                skippedNotWorthwhile++;
                if (opt.verbose)
                {
                    log << "Skipping bone \"" << job.name << "\" (interp#" << job.interpBlockIdx
                        << "): converting at M=" << m << " would grow the file by " << netDelta
                        << " byte(s) -- not worthwhile\n";
                }
                continue;
            }
            keptJobs.push_back(std::move(job));
            keptJobData.push_back(std::move(jobData[j]));
            keptJobM.push_back(m);
        }
        result.bonesSkippedNotWorthwhile = skippedNotWorthwhile;
        if (skippedNotWorthwhile && !opt.verbose)
        {
            log << "Skipped " << skippedNotWorthwhile
                << " bone(s) where B-spline conversion "
                   "wouldn't save space (use -v to see which, or --force-convert-all to convert anyway).\n";
        }
        jobs     = std::move(keptJobs);
        jobData  = std::move(keptJobData);
        jobM     = std::move(keptJobM);
        origKeyBytes = 0;
        for (auto &job : jobs) origKeyBytes += header.blockSizes[job.dataBlockIdx];

        if (jobs.empty())
        {
            log << "No bone remained eligible after the size-based conversion check "
                   "(every candidate would have gotten bigger, not smaller).\n";
            if (!opt.dryRun)
            {
                if (opt.inPath != opt.outPath)
                {
                    std::ofstream of(opt.outPath, std::ios::binary);
                    if (!of)
                        throw std::runtime_error("Can't open output file: " + opt.outPath);
                    of.write(reinterpret_cast<const char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
                    of.close();
                    log << "Copied through unchanged: " << opt.outPath << "\n";
                }
                result.wroteFile   = true;
                result.outputBytes = raw.size();
            }
            return result;
        }
    }

    if (opt.verbose)
    {
        std::set<int> distinctM(jobM.begin(), jobM.end());
        log << "Control-point count (M) per bone -- " << distinctM.size() << " distinct value(s) in use: ";
        for (int m : distinctM) log << m << " ";
        log << "\n";
        for (size_t j = 0; j < jobs.size(); ++j)
        {
            auto &job = jobs[j];
            log << "bone interp#" << job.interpBlockIdx << (job.name.empty() ? "" : " \"" + job.name + "\"")
                << " M=" << jobM[j] << ": ";
            if (job.translation.present)
                log << "trans err=" << job.translation.fit.maxError << "  ";
            if (job.rotation.present)
                log << "rot err=" << job.rotation.fit.maxError << "  ";
            if (job.scale.present)
                log << "scale err=" << job.scale.fit.maxError;
            log << "\n";
        }
    }

    result.bonesConverted = jobs.size();

    if (opt.dryRun)
    {
        log << "Dry run: " << jobs.size() << " bone channel(s) would be converted.\n";
        return result;
    }

    // ---- Assemble shared control-point pool + one basis block per distinct M ----
    std::vector<float> floatPool;
    std::vector<int16_t> compactPool;
    std::vector<std::vector<uint8_t>> newInterpBytes(jobs.size());

    int nextNewBlockIndex          = static_cast<int>(header.numBlocks);
    int sharedSplineDataFinalIndex = nextNewBlockIndex++;

    // Assign each distinct M value seen a basis-block index, in first-seen
    // order (order doesn't matter functionally, just needs to be stable).
    std::map<int, int> basisIndexForM; // M -> block index
    for (int m : jobM)
    {
        if (basisIndexForM.find(m) == basisIndexForM.end())
            basisIndexForM[m] = nextNewBlockIndex++;
    }

    for (size_t j = 0; j < jobs.size(); ++j)
    {
        auto &job = jobs[j];
        BSplineTransformInterpolatorOut out;
        out.compact   = opt.compact16;
        out.transform = job.interp.transform; // static fallback pose for any missing channel
        float t0 = 1e30f, t1 = -1e30f;
        if (job.translation.present)
        {
            t0 = std::min(t0, job.translation.startTime);
            t1 = std::max(t1, job.translation.stopTime);
        }
        if (job.rotation.present)
        {
            t0 = std::min(t0, job.rotation.startTime);
            t1 = std::max(t1, job.rotation.stopTime);
        }
        if (job.scale.present)
        {
            t0 = std::min(t0, job.scale.startTime);
            t1 = std::max(t1, job.scale.stopTime);
        }
        if (t0 > t1)
        {
            t0 = 0;
            t1 = 1;
        }
        out.startTime = t0;
        out.stopTime  = t1;

        out.translationHandle =
            appendChannel(job.translation, opt.compact16, floatPool, compactPool, out.translationOffset, out.translationHalfRange);
        out.rotationHandle =
            appendChannel(job.rotation, opt.compact16, floatPool, compactPool, out.rotationOffset, out.rotationHalfRange);
        out.scaleHandle =
            appendChannel(job.scale, opt.compact16, floatPool, compactPool, out.scaleOffset, out.scaleHalfRange);

        out.splineDataRef = sharedSplineDataFinalIndex;
        out.basisDataRef  = basisIndexForM[jobM[j]];

        newInterpBytes[j] = out.serialize();
    }

    BSplineData sharedData;
    sharedData.floatControlPoints   = floatPool;
    sharedData.compactControlPoints = compactPool;
    auto sharedDataBytes            = sharedData.serialize();

    // ---- Rebuild header block tables ----
    std::string interpTypeName = opt.compact16 ? "NiBSplineCompTransformInterpolator" : "NiBSplineTransformInterpolator";
    int interpTypeIdx          = header.findOrAddBlockType(interpTypeName);
    int basisTypeIdx           = header.findOrAddBlockType("NiBSplineBasisData");
    int splineDataTypeIdx      = header.findOrAddBlockType("NiBSplineData");

    uint32_t newNumBlocks = header.numBlocks + 1 /* shared NiBSplineData */
                           + static_cast<uint32_t>(basisIndexForM.size()) /* one NiBSplineBasisData per distinct M */;
    std::vector<uint16_t> newBlockTypeIndex = header.blockTypeIndex;
    std::vector<uint32_t> newBlockSizes     = header.blockSizes;
    newBlockTypeIndex.resize(newNumBlocks);
    newBlockSizes.resize(newNumBlocks);

    // Track, per original block index, replacement bytes (for modified
    // interpolator + emptied data blocks); everything else copies as-is.
    std::map<int, std::vector<uint8_t>> replacedBlockBytes;
    for (size_t j = 0; j < jobs.size(); ++j)
    {
        auto &job                                 = jobs[j];
        replacedBlockBytes[job.interpBlockIdx]     = newInterpBytes[j];
        newBlockTypeIndex[job.interpBlockIdx]      = static_cast<uint16_t>(interpTypeIdx);
        newBlockSizes[job.interpBlockIdx]          = static_cast<uint32_t>(newInterpBytes[j].size());

        replacedBlockBytes[job.dataBlockIdx] = TransformData::writeEmpty();
        newBlockSizes[job.dataBlockIdx]      = static_cast<uint32_t>(replacedBlockBytes[job.dataBlockIdx].size());
        // type index for the data block is left unchanged (still
        // NiTransformData, just empty).
    }
    newBlockTypeIndex[sharedSplineDataFinalIndex]     = static_cast<uint16_t>(splineDataTypeIdx);
    replacedBlockBytes[sharedSplineDataFinalIndex]    = sharedDataBytes;
    newBlockSizes[sharedSplineDataFinalIndex]         = static_cast<uint32_t>(sharedDataBytes.size());

    for (auto &[m, blockIdx] : basisIndexForM)
    {
        BSplineBasisData basis;
        basis.numControlPoints  = static_cast<uint32_t>(m);
        auto basisBytes         = basis.serialize();
        newBlockTypeIndex[blockIdx] = static_cast<uint16_t>(basisTypeIdx);
        replacedBlockBytes[blockIdx] = basisBytes;
        newBlockSizes[blockIdx]      = static_cast<uint32_t>(basisBytes.size());
    }

    header.numBlocks       = newNumBlocks;
    header.blockTypeIndex  = newBlockTypeIndex;
    header.blockSizes      = newBlockSizes;

    // ---- Serialize the output file ----
    ByteWriter out;
    writeHeader(out, header);
    for (uint32_t i = 0; i < newNumBlocks; ++i)
    {
        auto it = replacedBlockBytes.find(static_cast<int>(i));
        if (it != replacedBlockBytes.end())
        {
            out.raw(it->second);
        }
        else
        {
            out.raw(raw.data() + blockOffset[i], header.blockSizes[i]);
        }
    }
    if (trailerLen)
        out.raw(raw.data() + trailerStart, trailerLen);

    std::ofstream of(opt.outPath, std::ios::binary);
    if (!of)
        throw std::runtime_error("Can't open output file: " + opt.outPath);
    of.write(reinterpret_cast<const char *>(out.buf.data()), static_cast<std::streamsize>(out.buf.size()));
    of.close();
    result.wroteFile   = true;
    result.outputBytes = out.buf.size();

    size_t newControlBytes = sharedDataBytes.size();
    for (auto &[m, blockIdx] : basisIndexForM) newControlBytes += replacedBlockBytes[blockIdx].size();
    for (size_t j = 0; j < jobs.size(); ++j) newControlBytes += newInterpBytes[j].size();

    log << "Converted " << jobs.size() << " bone channel(s).\n"
        << "  Original key data:      " << origKeyBytes << " bytes\n"
        << "  New spline/control data: " << newControlBytes << " bytes\n"
        << "  File size: " << raw.size() << " -> " << out.buf.size() << " bytes ("
        << (100.0 * (double)out.buf.size() / (double)raw.size()) << "%)\n"
        << "Wrote " << opt.outPath << "\n";
    return result;
}

// ---------------------------------------------------------------- temp-file bridge
//
// convertFile() above is untouched, path-based logic (see the file header
// comment). This bridges it into bethutil's in-memory span-in/vector-out
// convention the same way btu::hkx::AnimExe::convert_impl already bridges a
// path-based external tool into that same convention: write the input to a
// private temp directory, run the real conversion, read the result back,
// clean up.
[[nodiscard]] auto make_working_dir() noexcept -> tl::expected<btu::Path, Error>
{
    static std::atomic<uint32_t> counter{0};
    const auto dir_name = "TEMPDIR_btu__kf" + std::to_string(counter++);
    auto dir_path        = fs::temp_directory_path() / dir_name;

    auto ec = std::error_code{};
    create_directory(dir_path, ec);
    if (ec)
        return tl::make_unexpected(Error(ec));

    return dir_path;
}

} // namespace

auto compress(std::span<const std::byte> input, const Settings &settings) noexcept
    -> tl::expected<std::vector<std::byte>, Error>
{
    const auto working_dir = make_working_dir();
    if (!working_dir)
        return tl::make_unexpected(working_dir.error());

    const auto input_path  = *working_dir / "input.kf";
    const auto output_path = *working_dir / "output.kf";

    const auto write_res = common::write_file(input_path, input);
    if (!write_res)
    {
        std::error_code ec{};
        remove_all(*working_dir, ec);
        return tl::make_unexpected(Error(KfErr::TempFileIo));
    }

    Options opt;
    opt.inPath             = input_path.string();
    opt.outPath            = output_path.string();
    opt.ratio               = settings.ratio;
    opt.useTolerance        = settings.use_tolerance;
    opt.tolerance            = settings.tolerance;
    opt.skip                 = settings.skip;
    opt.compact16            = settings.compact16;
    opt.forcedControlPoints = settings.control_points;
    opt.maxControlPoints    = settings.max_control_points;
    opt.noBlacklist          = settings.no_blacklist;
    opt.forceConvertAll     = settings.force_convert_all;
    for (const auto &name : settings.blacklist_add)
        opt.extraBlacklist.push_back(common::as_ascii_string(name));

    std::ostringstream log;
    auto convert_result = tl::expected<std::vector<std::byte>, Error>{};
    try
    {
        convertFile(opt, log);
        convert_result = common::read_file(output_path);
    }
    catch (const std::exception &)
    {
        // convertFile() throws std::runtime_error with a rich diagnostic
        // message on any format/validation failure - see its own throw
        // sites for what specifically can go wrong. That detail doesn't
        // survive into std::error_code (see error_code.hpp's comment), so
        // callers only get the coarse category here; nothing upstream of
        // this function currently surfaces exception text either (matches
        // every other bethutil module's expected<T, Error> convention).
        convert_result = tl::make_unexpected(Error(KfErr::UnsupportedFormat));
    }

    std::error_code ec{};
    remove_all(*working_dir, ec);

    return convert_result;
}

} // namespace btu::kf

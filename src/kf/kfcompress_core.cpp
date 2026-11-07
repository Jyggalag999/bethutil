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
// kfcompress core: bytes in, bytes out. See kfcompress_core.hpp.

#include "btu/kf/detail/kfcompress_core.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "btu/kf/detail/bspline.hpp"
#include "btu/kf/detail/dedupe_types.hpp"
#include "btu/kf/detail/niftypes.hpp"

using namespace nif;

namespace kfc {

// ---------------------------------------------------------------- name blacklist
//
// Mirrors the game/toolchain's own IsBlacklistedName() check: bones used for
// weapon attachment, IK end-effectors, or marked with "##" are excluded from
// lossy B-spline conversion by default, since those need exact (not
// approximated) transforms -- e.g. a weapon socket a few thousandths off
// is far more visible/gameplay-relevant than the same error on a spine bone.
static bool isBlacklistedName(const std::string& name, const std::vector<std::string>& extra) {
    // The first block (Weapon .. R Clavicle) matches the original vanilla
    // toolchain's own blacklist exactly. Everything after is this tool's
    // own addition, not part of that original list --
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
    //   - "Bip01 L Hand/Forearm/UpperArm/Clavicle": the original list only ever
    //     covers the right arm (the typical weapon-holding hand), but 1st-
    //     person reload animations routinely use the LEFT hand too (racking
    //     a slide, gripping a magazine/foregrip, holding the Pip-Boy on the
    //     wrist it's worn on) -- confirmed by real conversions showing up to
    //     0.019 rad rotation error on "Bip01 L Forearm" in a reload
    //     animation, at exactly the moment fingers need to align with a
    //     weapon part.
    static const char* kBuiltin[] = {
        "Weapon", "Bip01 WeaponBow", "B42AttachPointL", "ArrowNode",
        "Bip01 R Hand", "Bip01 R Forearm", "Bip01 R UpperArm", "Bip01 R Clavicle",
        "Bip01 L Hand", "Bip01 L Forearm", "Bip01 L UpperArm", "Bip01 L Clavicle",
    };
    for (auto* n : kBuiltin) if (name == n) return true;
    for (auto& n : extra) if (name == n) return true;
    if (name.find("##") != std::string::npos) return true;

    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower.find("camera") != std::string::npos) return true;

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
    for (const char* frag : {"shell", "bullet", "projectile", "casing", "round"}) {
        if (lower.find(frag) != std::string::npos) return true;
    }
    return false;
}

struct Quat { double w = 1, x = 0, y = 0, z = 0; };

static Quat eulerXYZToQuat(double rx, double ry, double rz) {
    // Extrinsic X-then-Y-then-Z: q = qz * qy * qx
    double cx = std::cos(rx * 0.5), sx = std::sin(rx * 0.5);
    double cy = std::cos(ry * 0.5), sy = std::sin(ry * 0.5);
    double cz = std::cos(rz * 0.5), sz = std::sin(rz * 0.5);
    Quat qx{cx, sx, 0, 0}, qy{cy, 0, sy, 0}, qz{cz, 0, 0, sz};
    auto mul = [](const Quat& a, const Quat& b) {
        return Quat{
            a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
            a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
        };
    };
    return mul(qz, mul(qy, qx));
}

// ---------------------------------------------------------------- sampling

static std::vector<double> decimateKeepEnds(const std::vector<double>& times, int skip) {
    std::vector<double> keep;
    if (times.empty()) return keep;
    for (size_t i = 0; i < times.size(); i += static_cast<size_t>(std::max(1, skip)))
        keep.push_back(times[i]);
    if (keep.back() != times.back()) keep.push_back(times.back());
    return keep;
}

// ---------------------------------------------------------------- conversion job

// A channel's control points as stored in the file (see quantise()).
struct Quantised {
    float offset = 0, halfRange = 1;
    std::vector<int16_t> q;         // compact16 only
    std::vector<double> decoded;    // [M * dim], as the engine reads the stored bytes
};

struct ChannelFit {
    bool present = false;
    int numComponents = 0;
    bspline::FitResult fit;
    float startTime = 0, stopTime = 1;
    Quantised quant;
    double worstError = 0; // budget-mode check: worst error / its limit (<= 1 passes)
};

struct BoneJob {
    int interpBlockIdx = -1;
    int dataBlockIdx = -1;
    std::string name;
    TransformInterpolator interp;
    ChannelFit translation, rotation, scale;
    // Union of the [start, stop] of every NiControllerSequence that references
    // this interpolator (see resolveBoneRange).
    double seqT0 = 0, seqT1 = 0;
    bool haveSeqRange = false;
    bool lossless = false; // left as plain keys: no fit met the budget / unusable source
    // Channels stored as the interpolator's static pose value (see markConstantChannel).
    bool staticCh[3] = {false, false, false};
    std::array<double, 4> staticVal[3] = {};
    double boneT0 = 0, boneT1 = 1; // fit range, used for an interpolator with no spline channel
};

// A handful of real mod files ship literal NaN/Inf keyframe values (found
// via a 14,313-file real batch: "Bip01 RPauldron" in one mod's idle
// animations has a NaN translation+rotation on its very first key --
// pre-existing garbage in that file, not something introduced here). NaN
// silently poisons every downstream computation (the least-squares fit,
// the min/max range used for --compact16's offset/halfRange), eventually
// surfacing as an infinite halfRange far from where the actual bad value
// was -- so check at the source instead of trying to catch it later.
static bool allFinite(const std::vector<std::vector<double>>& vals) {
    for (auto& row : vals)
        for (double v : row)
            if (!std::isfinite(v)) return false;
    return true;
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
static void computeBoneTimeRange(const TransformData& d, double& outT0, double& outT1) {
    double t0 = 1e30, t1 = -1e30;
    auto extend = [&](double t) { t0 = std::min(t0, t); t1 = std::max(t1, t); };
    for (auto& k : d.translations.keys) extend(k.time);
    if (d.hasRotation()) {
        if (!d.isXYZRotation()) {
            for (auto& k : d.quatKeys) extend(k.time);
        } else {
            for (int a = 0; a < 3; ++a)
                for (auto& k : d.xyzRotations[a].keys) extend(k.time);
        }
    }
    for (auto& k : d.scales.keys) extend(k.time);
    if (t0 > t1) { t0 = 0; t1 = 1; }
    outT0 = t0; outT1 = t1;
}

// The range every channel of a bone is fitted over, and the one written as the
// interpolator's start/stop: the sequence's [start, stop], not the bone's own
// first/last key. The engine clamps playback time to the sequence range only,
// then normalises with the interpolator's range WITHOUT clamping. A bone whose
// keys end early (or start late) would therefore be evaluated outside its
// parametric [0,1]: the cubic extrapolates past its last span, and before the
// start the decoder indexes control points ahead of the channel's own handle
// (another channel's data). Fitting over the full sequence range, with the
// first/last key value held out to both ends (see addHoldSamples), gives the
// same pose the uncompressed interpolator holds there. Falls back to the keys'
// own span when the sequence range is empty or inverted.
static void resolveBoneRange(const BoneJob& job, const TransformData& d, double& outT0, double& outT1) {
    if (job.haveSeqRange && job.seqT1 > job.seqT0) {
        outT0 = static_cast<float>(job.seqT0);
        outT1 = static_cast<float>(job.seqT1);
        return;
    }
    computeBoneTimeRange(d, outT0, outT1);
}

// ---------------------------------------------------------------- source curves, verification
//
// Every channel of a bone is read into a KeyCurve (times + 1/3/4-component
// values) and from it two sample sets over the bone's fit range [t0, t1]:
//   - fit samples: the keys inside the range (decimated by --skip) plus the
//     curve's value at t0 and t1 when no key sits there, normalised to [0,1];
//   - check samples: every key inside the range, the two end points, and the
//     midpoint of every gap shorter than kSnapGap, with the value the
//     uncompressed interpolator produces there (linear / slerp between keys,
//     held outside them). Gaps of kSnapGap or more are pose changes (snaps),
//     where a midpoint says nothing about the animation.
// The M search accepts a candidate only when the *quantised bytes* decode,
// the way the engine decodes them, to within budget at every check sample.

constexpr double kSnapGap = 0.2; // seconds

struct KeyCurve {
    int dim = 0;
    KeyType type = KeyType::LINEAR_KEY;
    bool unsupported = false;                 // key type the engine decode was not ported for (TBC)
    std::vector<double> t;                    // key times (Euler: union of the three axes' times)
    std::vector<std::array<double, 4>> v;     // key values
    std::vector<std::array<double, 4>> ta, tb; // QUADRATIC: first/second stored tangent; Bezier rotation: ta = intermediate
    bool euler = false;
    std::vector<KeyCurve> axes;               // Euler: X, Y, Z scalar curves, each with its own key times and type
    bool present() const { return !t.empty(); }
};

struct ChannelSamples {
    bool present = false;
    int dim = 0;
    size_t keyCount = 0;                              // keys inside the fit range
    std::vector<double> ts;                           // fit samples, normalised
    std::vector<std::vector<double>> vals;
    std::vector<double> checkT;                       // check samples, normalised
    std::vector<std::array<double, 4>> checkRef;      // unit quaternion for dim 4
    // Set (and `present` cleared) by markConstantChannel: every fit and check
    // sample is bit-identical, so the channel is stored as the interpolator's
    // static pose value instead of a spline.
    bool constant = false;
    std::array<double, 4> constValue{};
};

struct BoneSamples {
    ChannelSamples ch[3]; // translation, rotation, scale
};

static void normalise4(std::array<double, 4>& q) {
    double l = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (l > 0) for (double& x : q) x /= l;
}

// ---- the engine's quaternion helpers (w, x, y, z order), ported from its own
// NiQuaternion code. Its "Slerp" is not a true slerp: it is a normalised lerp
// whose weight is warped by a fitted polynomial (CounterWarp), and it takes no
// short path (the key loader aligns hemispheres once instead).

static double counterWarp(double t, double d) {
    double v3 = 1.0 - d * 0.8227968811988831;
    double v5 = v3 * v3 * 0.5854921936988831;
    return t * (v5 + v5 * t * (t + t - 3.0) + 1.0);
}

static std::array<double, 4> engineSlerp(double t, const std::array<double, 4>& a, const std::array<double, 4>& b) {
    double d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    double k = (t > 0.5) ? 1.0 - counterWarp(1.0 - t, d) : counterWarp(t, d);
    std::array<double, 4> r;
    for (int i = 0; i < 4; ++i) r[i] = a[i] + (b[i] - a[i]) * k;
    normalise4(r);
    return r;
}

static std::array<double, 4> quatMul(const std::array<double, 4>& a, const std::array<double, 4>& b) {
    return {a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]};
}

static std::array<double, 4> quatLog(const std::array<double, 4>& q) {
    double theta = q[0] <= -1.0 ? 3.14159265358979323846 : (q[0] >= 1.0 ? 0.0 : std::acos(q[0]));
    double s = std::sin(theta);
    double k = std::fabs(s) >= 0.001 ? theta / s : 1.0;
    return {0.0, q[1] * k, q[2] * k, q[3] * k};
}

static std::array<double, 4> quatExp(const std::array<double, 4>& v) {
    double len = std::sqrt(v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
    double s = std::sin(len);
    double k = std::fabs(s) >= 0.001 ? s / len : 1.0;
    return {std::cos(len), v[1] * k, v[2] * k, v[3] * k};
}

// Squad control point of key `cur` between its neighbours (the engine's
// NiQuaternion::Intermediate); end keys use themselves as the missing neighbour.
static std::array<double, 4> quatIntermediate(const std::array<double, 4>& prev, const std::array<double, 4>& cur,
                                              const std::array<double, 4>& next) {
    std::array<double, 4> conj = {cur[0], -cur[1], -cur[2], -cur[3]};
    auto l1 = quatLog(quatMul(conj, next));
    auto l2 = quatLog(quatMul(conj, prev));
    std::array<double, 4> s = {-0.25 * (l1[0] + l2[0]), -0.25 * (l1[1] + l2[1]), -0.25 * (l1[2] + l2[2]), -0.25 * (l1[3] + l2[3])};
    return quatMul(cur, quatExp(s));
}

static std::array<double, 4> quatSquad(double t, const std::array<double, 4>& q0, const std::array<double, 4>& a,
                                       const std::array<double, 4>& b, const std::array<double, 4>& q1) {
    auto s1 = engineSlerp(t, q0, q1);
    auto s2 = engineSlerp(t, a, b);
    return engineSlerp((1.0 - t) * (t + t), s1, s2);
}

static double hermite(double t, double v0, double m0, double v1, double m1) {
    double d = v1 - v0;
    return t * (m0 + ((m0 + m1 - 2.0 * d) * t + (3.0 * d - (m1 + 2.0 * m0))) * t) + v0;
}

// Value of the uncompressed channel at time t, the way the engine plays it:
// the first/last key's value outside the key range; inside, the segment whose
// end key is the first at or after t (so t on a key gives that key's value),
// interpolated by the channel's key type. Euler rotations evaluate each axis
// on its own keys and then build the quaternion.
static std::array<double, 4> curveAt(const KeyCurve& c, double t) {
    if (c.euler) {
        double e[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a)
            if (c.axes[a].present()) e[a] = curveAt(c.axes[a], t)[0];
        Quat q = eulerXYZToQuat(e[0], e[1], e[2]);
        return {q.w, q.x, q.y, q.z};
    }
    auto done = [&](std::array<double, 4> r) { if (c.dim == 4) normalise4(r); return r; };
    const size_t n = c.t.size();
    if (n == 1 || t <= c.t.front()) return done(c.v.front());
    if (t >= c.t.back()) return done(c.v.back());
    size_t idx = static_cast<size_t>(std::lower_bound(c.t.begin(), c.t.end(), t) - c.t.begin());
    size_t i = idx - 1;
    double span = c.t[idx] - c.t[i];
    double u = span > 0 ? (t - c.t[i]) / span : 0.0;
    if (c.type == KeyType::CONST_KEY) return done(u >= 1.0 ? c.v[idx] : c.v[i]);
    std::array<double, 4> r{};
    if (c.dim == 4) {
        if (c.type == KeyType::QUADRATIC_KEY) return quatSquad(u, c.v[i], c.ta[i], c.ta[idx], c.v[idx]);
        return engineSlerp(u, c.v[i], c.v[idx]);
    }
    for (int k = 0; k < c.dim; ++k) {
        if (c.type == KeyType::QUADRATIC_KEY) r[k] = hermite(u, c.v[i][k], c.tb[i][k], c.v[idx][k], c.ta[idx][k]);
        else r[k] = c.v[i][k] + (c.v[idx][k] - c.v[i][k]) * u;
    }
    return r;
}

static bool curveFinite(const KeyCurve& c) {
    if (c.euler) {
        for (auto& a : c.axes) if (!curveFinite(a)) return false;
        return true;
    }
    for (size_t i = 0; i < c.t.size(); ++i) {
        if (!std::isfinite(c.t[i])) return false;
        for (int k = 0; k < c.dim; ++k) {
            if (!std::isfinite(c.v[i][k])) return false;
            if (c.type == KeyType::QUADRATIC_KEY && c.dim != 4 && !(std::isfinite(c.ta[i][k]) && std::isfinite(c.tb[i][k]))) return false;
        }
    }
    return true;
}

static KeyCurve translationCurve(const KeyGroup<Vector3>& g) {
    KeyCurve c; c.dim = 3; c.type = g.interpolation;
    c.unsupported = g.present() && g.interpolation == KeyType::TBC_KEY;
    for (auto& k : g.keys) {
        c.t.push_back(k.time);
        c.v.push_back({k.value.x, k.value.y, k.value.z, 0});
        c.ta.push_back({k.tanA.x, k.tanA.y, k.tanA.z, 0});
        c.tb.push_back({k.tanB.x, k.tanB.y, k.tanB.z, 0});
    }
    return c;
}

static KeyCurve scaleCurve(const KeyGroup<float>& g) {
    KeyCurve c; c.dim = 1; c.type = g.interpolation;
    c.unsupported = g.present() && g.interpolation == KeyType::TBC_KEY;
    for (auto& k : g.keys) {
        c.t.push_back(k.time);
        c.v.push_back({k.value, 0, 0, 0});
        c.ta.push_back({k.tanA, 0, 0, 0});
        c.tb.push_back({k.tanB, 0, 0, 0});
    }
    return c;
}

static KeyCurve rotationCurve(const TransformData& d) {
    KeyCurve c; c.dim = 4;
    if (!d.hasRotation()) return c;
    if (d.isXYZRotation()) {
        c.euler = true;
        std::set<float> times;
        for (int a = 0; a < 3; ++a) {
            c.axes.push_back(scaleCurve(d.xyzRotations[a]));
            if (c.axes.back().unsupported) c.unsupported = true;
            for (auto& k : d.xyzRotations[a].keys) times.insert(k.time);
        }
        c.t.assign(times.begin(), times.end());
        return c;
    }
    c.type = d.rotationType;
    c.unsupported = d.rotationType == KeyType::TBC_KEY;
    for (auto& k : d.quatKeys) { c.t.push_back(k.time); c.v.push_back({k.value.w, k.value.x, k.value.y, k.value.z}); }
    // The key loader aligns every key to the hemisphere of the one before it
    // (shortest path), once, before any interpolation.
    for (size_t i = 1; i < c.v.size(); ++i) {
        double dot = 0;
        for (int k = 0; k < 4; ++k) dot += c.v[i-1][k] * c.v[i][k];
        if (dot < 0) for (double& x : c.v[i]) x = -x;
    }
    if (c.type == KeyType::QUADRATIC_KEY && c.v.size() >= 2) {
        const size_t n = c.v.size();
        c.ta.resize(n);
        c.ta[0] = quatIntermediate(c.v[0], c.v[0], c.v[1]);
        for (size_t i = 1; i + 1 < n; ++i) c.ta[i] = quatIntermediate(c.v[i-1], c.v[i], c.v[i+1]);
        c.ta[n-1] = quatIntermediate(c.v[n-2], c.v[n-1], c.v[n-1]);
    }
    return c;
}

static ChannelSamples makeSamples(const KeyCurve& c, int skip, double t0, double t1) {
    ChannelSamples s;
    if (!c.present()) return s;
    const double eps = 1e-7;
    std::vector<double> times; // key times inside the range, end points included
    for (double t : c.t)
        if (t >= t0 - eps && t <= t1 + eps) times.push_back(std::clamp(t, t0, t1));
    if (times.empty() || times.front() > t0 + eps) times.insert(times.begin(), t0);
    if (times.back() < t1 - eps) times.push_back(t1);

    auto norm = [&](double t) { return (t1 > t0) ? std::clamp((t - t0) / (t1 - t0), 0.0, 1.0) : 0.0; };
    auto row = [&](const std::array<double, 4>& a) { return std::vector<double>(a.begin(), a.begin() + c.dim); };
    s.present = true; s.dim = c.dim;
    s.keyCount = times.size();
    // Fit to what is verified: the keys AND the midpoints of short gaps. Fitting
    // keys alone leaves each midpoint to whatever the smooth curve between two
    // keys happens to do, which on jittery high-rate motion never lands on the
    // straight segment the uncompressed interpolator plays, however large M gets.
    auto kept = decimateKeepEnds(times, skip);
    for (size_t i = 0; i < kept.size(); ++i) {
        s.ts.push_back(norm(kept[i])); s.vals.push_back(row(curveAt(c, kept[i])));
        if (c.dim == 4 && s.vals.size() >= 2) { // keep consecutive fit quaternions in one hemisphere
            auto& p = s.vals[s.vals.size() - 2]; auto& q = s.vals.back();
            if (p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3] < 0) for (double& x : q) x = -x;
        }
        if (i + 1 < kept.size() && kept[i + 1] - kept[i] < kSnapGap) {
            double mid = 0.5 * (kept[i] + kept[i + 1]);
            s.ts.push_back(norm(mid)); s.vals.push_back(row(curveAt(c, mid)));
            if (c.dim == 4) {
                auto& p = s.vals[s.vals.size() - 2]; auto& q = s.vals.back();
                if (p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3] < 0) for (double& x : q) x = -x;
            }
        }
    }
    for (size_t i = 0; i < times.size(); ++i) {
        s.checkT.push_back(norm(times[i])); s.checkRef.push_back(curveAt(c, times[i]));
        if (i + 1 < times.size() && times[i + 1] - times[i] < kSnapGap) {
            double mid = 0.5 * (times[i] + times[i + 1]);
            s.checkT.push_back(norm(mid)); s.checkRef.push_back(curveAt(c, mid));
        }
    }
    return s;
}

// True when the curve is plain linear (every axis for Euler rotation): between
// keys it can only move on the straight segment, so equal key values mean it
// never leaves that value. Tangent-carrying key types are excluded because
// identical key values do not stop them bulging between the keys.
static bool curveIsLinear(const KeyCurve& c) {
    if (c.euler) {
        for (const auto& a : c.axes) if (a.type != KeyType::LINEAR_KEY) return false;
        return true;
    }
    return c.type == KeyType::LINEAR_KEY;
}

// A channel whose every fit and check sample is bit-identical never moves: it
// needs no control points. It is stored as the interpolator's static pose value
// (handle left at "no data"), which is exact and free. Exact equality only, no
// epsilon, so nothing that moves at all is ever collapsed.
static void markConstantChannel(const KeyCurve& c, ChannelSamples& s) {
    if (!s.present || s.vals.empty() || !curveIsLinear(c)) return;
    for (size_t i = 1; i < s.vals.size(); ++i) if (s.vals[i] != s.vals[0]) return;
    for (size_t i = 1; i < s.checkRef.size(); ++i) if (s.checkRef[i] != s.checkRef[0]) return;
    s.constant = true;
    for (int k = 0; k < s.dim; ++k) s.constValue[k] = s.vals[0][k];
    s.present = false;
}

// Builds the fit/check samples of every channel of one bone. False when a
// source key is NaN/Inf, or a channel uses TBC keys (whose engine playback is
// not ported here): the bone is then left as plain keys.
static bool buildBoneSamples(const TransformData& d, int skip, double t0, double t1, bool collapseConstant, BoneSamples& out) {
    KeyCurve cs[3] = {translationCurve(d.translations), rotationCurve(d), scaleCurve(d.scales)};
    for (int i = 0; i < 3; ++i) {
        if (cs[i].unsupported || !curveFinite(cs[i])) return false;
        out.ch[i] = makeSamples(cs[i], skip, t0, t1);
        if (collapseConstant) markConstantChannel(cs[i], out.ch[i]);
    }
    return true;
}

// ---------------------------------------------------------------- engine decode

// Span and weights of the engine's uniform clamped cubic basis for M control
// points at normalised time t: iMax = (int)((M-3)*t)+3 (M-1 for t >= 1), the
// four weights of control points iMax-3..iMax.
static void engineBasis(int M, double t, int& iMin, double w[4]) {
    int iMax = (t >= 1.0) ? M - 1 : static_cast<int>((M - 3) * t) + 3;
    iMin = iMax - 3;
    const int n = M - 3;
    auto knot = [&](int i) { return static_cast<double>(std::clamp(i - 3, 0, n)) / n; };
    double left[4] = {0, 0, 0, 0}, right[4] = {0, 0, 0, 0};
    w[0] = 1; w[1] = w[2] = w[3] = 0;
    for (int j = 1; j <= 3; ++j) {
        left[j] = t - knot(iMax + 1 - j);
        right[j] = knot(iMax + j) - t;
        double saved = 0;
        for (int r = 0; r < j; ++r) {
            double den = right[r + 1] + left[j - r];
            double tmp = den != 0 ? w[r] / den : 0.0;
            w[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        w[j] = saved;
    }
}

// The one place that defines how control points become stored values:
// appendChannel writes `q` / the floats, the verifier decodes `decoded`.
// compact16 uses the float-rounded offset/halfRange that are written to the
// file (the engine decodes with those, not the doubles), rounding halfRange
// up so no control point clips.
static Quantised quantise(const std::vector<std::vector<double>>& cps, bool compact) {
    Quantised r;
    if (!compact) {
        for (auto& row : cps) for (double v : row) r.decoded.push_back(static_cast<double>(static_cast<float>(v)));
        return r;
    }
    double lo = 1e300, hi = -1e300;
    for (auto& row : cps) for (double v : row) { lo = std::min(lo, v); hi = std::max(hi, v); }
    double offset = (lo + hi) * 0.5;
    double halfRange = std::max({std::fabs(hi - offset), std::fabs(lo - offset), 1e-6});
    r.offset = static_cast<float>(offset);
    r.halfRange = static_cast<float>(halfRange);
    double needed = std::max({std::fabs(hi - r.offset), std::fabs(lo - r.offset), 1e-6});
    while (static_cast<double>(r.halfRange) < needed) r.halfRange = std::nextafter(r.halfRange, std::numeric_limits<float>::infinity());
    for (auto& row : cps) {
        for (double v : row) {
            double n = (v - r.offset) / r.halfRange; // [-1, 1]
            long q = std::clamp<long>(std::lround(n * 32767.0), -32767, 32767);
            r.q.push_back(static_cast<int16_t>(q));
            r.decoded.push_back(static_cast<double>(q) / 32767.0 * static_cast<double>(r.halfRange) + static_cast<double>(r.offset));
        }
    }
    return r;
}

// Translation at the two ends of the sequence is held to a much tighter limit
// than the budget: the engine derives a locomotion animation's walking speed
// from the root's position at sequence begin and end, divided by the duration,
// so an end error that is harmless as a pose becomes a speed error (0.006 units
// on a 1 s clip at 5 u/s is 0.1%).
constexpr double kEndTranslationLimit = 0.001; // game units

// Worst error of a quantised channel over its check samples, as a multiple of
// the limit that applies to each sample (the budget; the tighter end limit for
// translation at t = 0 and t = 1). <= 1 passes.
static double worstChannelError(const ChannelSamples& s, const Quantised& qz, int M, double budget) {
    double worst = 0;
    for (size_t i = 0; i < s.checkT.size(); ++i) {
        int iMin; double w[4];
        engineBasis(M, s.checkT[i], iMin, w);
        std::array<double, 4> v{};
        for (int j = 0; j < 4; ++j) {
            int idx = iMin + j;
            if (idx < 0 || idx >= M) return std::numeric_limits<double>::infinity(); // engine would read outside the channel
            for (int c = 0; c < s.dim; ++c) v[c] += w[j] * qz.decoded[static_cast<size_t>(idx) * s.dim + c];
        }
        const auto& ref = s.checkRef[i];
        double e;
        if (s.dim == 4) {
            normalise4(v); // the engine renormalises after decoding
            double dot = std::fabs(v[0]*ref[0] + v[1]*ref[1] + v[2]*ref[2] + v[3]*ref[3]);
            e = 2.0 * std::acos(std::min(1.0, dot)) * (180.0 / 3.14159265358979323846);
        } else {
            double sq = 0;
            for (int c = 0; c < s.dim; ++c) sq += (v[c] - ref[c]) * (v[c] - ref[c]);
            e = std::sqrt(sq);
        }
        double limit = budget;
        if (s.dim == 3 && (s.checkT[i] <= 0.0 || s.checkT[i] >= 1.0)) limit = std::min(budget, kEndTranslationLimit);
        worst = std::max(worst, e / limit);
    }
    return worst;
}

// ---------------------------------------------------------------- bone fit

// Control-point count the ratio rule gives: duration-driven, not key-count-
// driven (M = round(0.5 * ratio * sampleCount) for a ~61 Hz sampling of the
// range), so a sparse channel still gets as many control points as a dense
// one over the same time; the floor keeps short snappy clips from starving.
// Only a starting guess in budget mode.
static int ratioControlPoints(double durationSeconds, double ratio, int maxControlPoints, int minControlPoints) {
    int minM = bspline::kDegree + 1;
    int ratioMaxM = (maxControlPoints > 0) ? maxControlPoints : std::numeric_limits<int>::max();
    if (minControlPoints > 0) minM = std::max(minM, std::min(minControlPoints, ratioMaxM));
    int sampleCount = static_cast<int>(std::lround(durationSeconds * 61.000004)) + 1;
    int m = static_cast<int>(std::lround(0.5 * ratio * static_cast<double>(sampleCount)));
    return std::clamp(m, minM, ratioMaxM);
}

static ChannelFit fitChannel(const ChannelSamples& s, int M, bool compact, double t0, double t1) {
    ChannelFit cf;
    if (!s.present) return cf;
    cf.fit = bspline::fitBSpline(s.ts, s.vals, M);
    cf.present = true;
    cf.numComponents = s.dim;
    cf.startTime = static_cast<float>(t0);
    cf.stopTime = static_cast<float>(t1);
    cf.quant = quantise(cf.fit.controlPoints, compact);
    return cf;
}

// Fits all channels at one M; with budgets, also checks the quantised result.
// A channel whose budget is <= 0 is not checked.
static bool fitBoneAtM(const BoneSamples& bs, int M, bool compact, double t0, double t1,
                       const double budget[3], ChannelFit out[3]) {
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        out[i] = fitChannel(bs.ch[i], M, compact, t0, t1);
        if (!out[i].present) continue;
        if (!allFinite(out[i].fit.controlPoints)) return false;
        if (budget[i] > 0) {
            out[i].worstError = worstChannelError(bs.ch[i], out[i].quant, M, budget[i]);
            if (!(out[i].worstError <= 1.0)) ok = false;
        }
    }
    return ok;
}

// Smallest verified M for one bone, starting at the ratio guess m0. Passing at
// m0: look for a cheaper M below it; failing: gallop up. Error does not fall
// monotonically with M, so no result is trusted unless it was itself verified.
// Returns false (bone stays lossless) when nothing up to maxM passes.
static bool searchBoneM(const BoneSamples& bs, int m0, int maxM, bool compact, double t0, double t1,
                        const double budget[3], ChannelFit best[3], int& bestM) {
    const int minM = bspline::kDegree + 1;
    ChannelFit cand[3];
    bool have = false;
    auto attempt = [&](int M) {
        if (!fitBoneAtM(bs, M, compact, t0, t1, budget, cand)) return false;
        for (int i = 0; i < 3; ++i) best[i] = cand[i];
        bestM = M; have = true;
        return true;
    };
    m0 = std::max(m0, minM);
    int lo, hi; // lo fails (or is the floor), hi passes
    if (attempt(m0)) {
        hi = m0;
        lo = std::max(minM, m0 / 4);
        if (lo >= hi || attempt(lo)) return true;
    } else {
        lo = m0; hi = 0;
        int m = m0;
        while (m < maxM) {
            m = std::min(maxM, std::max(m + 1, static_cast<int>(std::ceil(m * 1.5))));
            if (attempt(m)) { hi = m; break; }
            lo = m;
        }
        if (!hi) return false;
    }
    while (hi - lo > 1) {
        int mid = lo + (hi - lo) / 2;
        if (attempt(mid)) hi = mid; else lo = mid;
    }
    return have;
}

// True when every control point of every present channel is finite. A
// degenerate least-squares solve could in principle produce NaN/Inf even from
// finite keys (those are rejected up front, see allFinite), and std::lround on
// a non-finite value is undefined behaviour in appendChannel's compact path.
static bool jobControlPointsFinite(const BoneJob& job) {
    for (const ChannelFit* cf : {&job.translation, &job.rotation, &job.scale}) {
        if (!cf->present) continue;
        if (!allFinite(cf->fit.controlPoints)) return false;
    }
    return true;
}

// Appends one channel's control points into the shared pool (float or
// compact int16), returning the element offset ("Handle") of its first
// control point. The stored values come from cf.quant, the same ones the
// budget check decoded.
// ---- Block removal support ----
// After conversion the old NiTransformData blocks are unreachable. Dropping
// them (instead of leaving 12-byte placeholders) means compacting the block
// index space, so EVERY block reference in the file has to be rewritten.
// That is only safe for block types whose reference layout we know exactly,
// so the whitelist below is deliberately small: anything else in the file
// makes us keep the placeholders (no guessing about layouts we haven't seen).
static bool blockTypeRefsKnown(const std::string& t) {
    static const std::set<std::string> known = {
        // carry references
        "NiControllerSequence", "NiTransformInterpolator", "NiFloatInterpolator",
        "NiBoolInterpolator", "NiPoint3Interpolator",
        "NiBSplineTransformInterpolator", "NiBSplineCompTransformInterpolator",
        "NiBSplineFloatInterpolator", "NiBSplineCompFloatInterpolator",
        "NiBSplineCompPoint3Interpolator",
        // carry none (name string index / plain numeric payloads only)
        "NiTextKeyExtraData", "NiTransformData", "NiKeyframeData", "NiFloatData",
        "NiPosData", "NiBoolData", "NiBSplineData", "NiBSplineBasisData",
        "NiStringPalette"};
    return known.count(t) != 0;
}

// Rewrites one int32 block reference in place through newIndex (-1 = removed).
static bool patchRef(std::vector<uint8_t>& b, size_t off, const std::vector<int32_t>& newIndex) {
    if (off + 4 > b.size()) return false;
    int32_t r;
    std::memcpy(&r, b.data() + off, 4);
    if (r < 0) return true; // "no reference" stays as-is
    if (static_cast<size_t>(r) >= newIndex.size() || newIndex[r] < 0) return false; // dangling / removed target
    r = newIndex[r];
    std::memcpy(b.data() + off, &r, 4);
    return true;
}

// Rewrites every block reference in `b` through newIndex (-1 = removed).
// Returns false if the block cannot be remapped with certainty; the caller
// then falls back to keeping placeholders.
static bool remapBlockRefs(const std::string& type, std::vector<uint8_t>& b,
                           const std::vector<int32_t>& newIndex, uint32_t bsVersion) {
    if (type == "NiControllerSequence") {
        try {
            ByteReader r(b.data(), b.size());
            ControllerSequence cs = ControllerSequence::read(r, bsVersion, static_cast<uint32_t>(b.size()));
            ByteWriter check;
            cs.write(check);
            if (check.buf != b) return false; // our model of the block loses something
            auto map = [&](int32_t& v) {
                if (v < 0) return true;
                if (static_cast<size_t>(v) >= newIndex.size() || newIndex[v] < 0) return false;
                v = newIndex[v];
                return true;
            };
            for (auto& cb : cs.blocks) if (!map(cb.interpolatorRef) || !map(cb.controllerRef)) return false;
            if (!map(cs.textKeysRef) || !map(cs.managerRef) || !map(cs.animNotesRef)) return false;
            for (auto& v : cs.animNoteArrayRefs) if (!map(v)) return false;
            ByteWriter w;
            cs.write(w);
            b = std::move(w.buf);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }
    // Interpolators whose only reference is the trailing data ref.
    if (type == "NiTransformInterpolator") return b.size() == 36 && patchRef(b, 32, newIndex);
    if (type == "NiFloatInterpolator")     return b.size() == 8  && patchRef(b, 4, newIndex);
    if (type == "NiBoolInterpolator")      return b.size() == 5  && patchRef(b, 1, newIndex);
    if (type == "NiPoint3Interpolator")    return b.size() == 16 && patchRef(b, 12, newIndex);
    // NiBSplineInterpolator base: startTime, stopTime, splineDataRef, basisDataRef.
    if (type.compare(0, 8, "NiBSplin") == 0 && type.find("Interpolator") != std::string::npos)
        return b.size() >= 16 && patchRef(b, 8, newIndex) && patchRef(b, 12, newIndex);
    return true; // whitelisted reference-free block
}

static uint32_t appendChannel(const ChannelFit& cf, bool compact,
                               std::vector<float>& floatPool,
                               std::vector<int16_t>& compactPool,
                               float& outOffset, float& outHalfRange) {
    outOffset = 0; outHalfRange = 1;
    if (!cf.present) return kNoHandle;

    // The engine treats a handle of exactly 0xFFFF as "channel absent" and skips
    // it, so a real channel must never start at pool element 65535: push one
    // padding element first (an unreferenced slot).
    if (compact) { if (compactPool.size() == kNoHandle) compactPool.push_back(0); }
    else if (floatPool.size() == kNoHandle) floatPool.push_back(0.0f);

    if (!compact) {
        uint32_t handle = static_cast<uint32_t>(floatPool.size());
        for (auto& row : cf.fit.controlPoints)
            for (double v : row) floatPool.push_back(static_cast<float>(v));
        return handle;
    }
    outOffset = cf.quant.offset;
    outHalfRange = cf.quant.halfRange;
    uint32_t handle = static_cast<uint32_t>(compactPool.size());
    compactPool.insert(compactPool.end(), cf.quant.q.begin(), cf.quant.q.end());
    return handle;
}

// ---------------------------------------------------------------- convertFile
//
// Does the actual work for one file. Writes progress/results into `log`
// rather than directly to cout/cerr so batch mode can run many of these
// concurrently (one per worker thread) and print each file's report as an
// atomic block afterwards, instead of interleaving output across threads.
// Throws std::runtime_error on unrecoverable errors (bad format, I/O
// failure, etc.), same as before -- callers decide whether that aborts the
// whole run (single-file mode) or is caught per-file (batch mode).

// ---------------------------------------------------------------- compressBytes

Result compressBytes(std::vector<uint8_t> input, const Options& opt, std::ostream& log) {
    Result result;

    std::vector<uint8_t> raw = std::move(input);
    result.inputBytes = raw.size();

    // ---- Optional lossless pre-pass: remove excess repeated keys before
    // the (lossy) B-spline conversion below ever sees the file. This is the
    // same reduction kfdedupe does standalone (see dedupe_types.hpp) --
    // folded in here so one invocation can do both instead of requiring a
    // separate kfdedupe run first. Applied unconditionally to every
    // NiTransformData block, including bones the B-spline blacklist would
    // skip: it's lossless, so there's no bone that needs protecting from it.
    // It must NOT feed the B-spline fit below, though: the fit is a least-
    // squares solve sampled only at key times, so a held pose that dedupe
    // collapses to its two endpoint keys leaves the fit with nothing
    // constraining the control points across the hold. The roughness
    // penalty then carries the slopes going into and out of the hold
    // straight through it, and the curve bulges far off the held value
    // (seen on a 12k-file mod batch: a 2s thumb hold decoded to a quat
    // component of -16, a neck translation off by 22 units). Losslessness
    // only holds for linear interpolation between keys, which is exactly
    // what the fit isn't doing. So the original bytes are kept in `fitRaw`
    // and every bone's fit input is read from there; the deduped `raw` is
    // only used for what gets written, i.e. bones left as plain keys.
    std::vector<uint8_t> fitRaw;
    if (opt.dedupe) {
        kfdedupe::DedupePassResult pass = kfdedupe::dedupeFileBytes(raw, opt.dedupeEpsilon);
        if (pass.changed) {
            result.dedupeBlocksTouched = pass.blocksTouched;
            result.dedupeKeysRemoved = pass.keysRemoved;
            if (opt.verbose) {
                log << "Dedupe pre-pass: removed " << pass.keysRemoved << " key(s) across "
                    << pass.blocksTouched << " block(s) (" << raw.size() << " -> " << pass.bytes.size() << " bytes)\n";
                for (auto& line : pass.verboseLines) log << "  " << line << "\n";
            }
            // The pre-dedupe bytes become the fit source; `raw` takes the
            // deduped bytes. Moved, not copied: nothing reads the old `raw`
            // after this point except through `fitRaw`.
            if (!opt.dedupeOnly) fitRaw = std::move(raw);
            raw = std::move(pass.bytes);
        }
    }

    // ---- --dedupe-only: stop here, matching standalone kfdedupe exactly.
    // No NIF header/block parsing happens at all in this mode -- the dedupe
    // pass above already did all the work; this just reports and writes it,
    // the same as kfdedupe.cpp's own convertFile.
    if (opt.dedupeOnly) {
        if (result.dedupeBlocksTouched == 0) {
            log << "No repeated keys found to remove.\n";
            if (!opt.dryRun) {
                result.outputBytes = raw.size();
                result.bytes = std::move(raw);
                result.wroteFile = true;
            }
            return result;
        }
        if (opt.dryRun) {
            log << "Dry run: " << result.dedupeBlocksTouched << " block(s), "
                << result.dedupeKeysRemoved << " key(s) would be removed.\n";
            return result;
        }
        result.outputBytes = raw.size();
        result.bytes = std::move(raw);
        result.modified = true;
        result.wroteFile = true;
        log << "Deduped " << result.dedupeBlocksTouched << " block(s), removed "
            << result.dedupeKeysRemoved << " key(s).\n"
            << "  File size: " << result.inputBytes << " -> " << result.outputBytes << " bytes ("
            << (100.0 * (double)result.outputBytes / (double)result.inputBytes) << "%)\n";
        return result;
    }

    ByteReader hr(raw);
    Header header = readHeader(hr);
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
    for (uint32_t i = 0; i < header.numBlocks; ++i) {
        if (header.blockTypeIndex[i] >= header.blockTypes.size()) {
            throw FormatError(
                "Block " + std::to_string(i) + " has an out-of-range block-type index "
                "(" + std::to_string(header.blockTypeIndex[i]) + " >= " +
                std::to_string(header.blockTypes.size()) + " known types) -- file is "
                "corrupt, truncated, or not actually this NIF version despite its extension.");
        }
    }

    // Compute per-block byte offsets within the file (blocks are
    // sequential right after the header, sized per header.blockSizes).
    std::vector<size_t> blockOffset(header.numBlocks);
    size_t cursor = hr.pos;
    for (uint32_t i = 0; i < header.numBlocks; ++i) {
        blockOffset[i] = cursor;
        cursor += header.blockSizes[i];
        // A block whose declared offset+size runs past the actual file
        // means every ByteReader constructed over that block's claimed
        // byte range would think it has more valid memory than it really
        // does (ByteReader::need() only checks against the *claimed*
        // size passed to it, not the real buffer bounds) -- a heap
        // buffer over-read. Must hard-fail here, not just warn, and
        // before any block gets read.
        if (cursor > raw.size()) {
            throw FormatError(
                "Block " + std::to_string(i) + "'s declared size runs past the end of "
                "the file (file is truncated or the block-size table is corrupt).");
        }
    }
    if (cursor != raw.size()) {
        log << "Warning: " << (raw.size() - cursor)
            << " trailing bytes after the last declared block; keeping them as-is.\n";
    }
    size_t trailerStart = cursor, trailerLen = raw.size() - cursor;

    // Where each block lives in the bytes the fit reads from (see the
    // dedupe pre-pass comment above). Dedupe only rewrites NiTransformData
    // blocks in place -- never adds, removes, or reorders blocks -- so block
    // indices are shared and only the offsets/sizes differ. Without a
    // dedupe change this is just `raw` itself.
    const uint8_t* fitBase = raw.data();
    std::vector<size_t> fitBlockOffset = blockOffset;
    std::vector<uint32_t> fitBlockSizes = header.blockSizes;
    if (!fitRaw.empty()) {
        ByteReader fhr(fitRaw);
        Header fitHeader = readHeader(fhr);
        if (fitHeader.numBlocks != header.numBlocks)
            throw std::runtime_error("Dedupe pre-pass changed the block count; refusing to fit against mismatched data.");
        size_t fc = fhr.pos;
        for (uint32_t i = 0; i < fitHeader.numBlocks; ++i) {
            fitBlockOffset[i] = fc;
            fc += fitHeader.blockSizes[i];
        }
        if (fc > fitRaw.size())
            throw FormatError("Original file's block-size table runs past the end of the file.");
        fitBase = fitRaw.data();
        fitBlockSizes = fitHeader.blockSizes;
    }

    auto blockType = [&](int idx) -> const std::string& { return header.blockTypes[header.blockTypeIndex[idx]]; };

    auto resolveString = [&](int32_t idx) -> std::string {
        if (idx < 0 || idx >= static_cast<int32_t>(header.strings.size())) return "";
        return header.strings[idx];
    };

    // ---- Find every NiControllerSequence, and every NiTransformInterpolator
    //      it references, and build one BoneJob per unique interpolator. ----
    std::vector<BoneJob> jobs;
    std::set<int> seenInterp;
    std::vector<int> seqIndices;
    size_t skippedByBlacklist = 0;
    for (uint32_t i = 0; i < header.numBlocks; ++i)
        if (blockType(i) == "NiControllerSequence") seqIndices.push_back(static_cast<int>(i));

    // Sequence [start, stop] per interpolator block (union if several sequences share one).
    std::map<int, std::pair<double, double>> seqRangeOf;

    for (int seqIdx : seqIndices) {
        ByteReader sr(raw.data() + blockOffset[seqIdx], header.blockSizes[seqIdx]);
        ControllerSequence seq = ControllerSequence::read(sr, bsVersion, header.blockSizes[seqIdx]);
        for (auto& cb : seq.blocks) {
            int interpIdx = cb.interpolatorRef;
            if (interpIdx < 0 || interpIdx >= static_cast<int>(header.numBlocks)) continue;
            if (blockType(interpIdx) != "NiTransformInterpolator") continue; // already bspline, or other type
            {
                auto r = seqRangeOf.emplace(interpIdx, std::pair<double, double>(seq.startTime, seq.stopTime));
                if (!r.second) {
                    r.first->second.first = std::min<double>(r.first->second.first, seq.startTime);
                    r.first->second.second = std::max<double>(r.first->second.second, seq.stopTime);
                }
            }
            if (seenInterp.count(interpIdx)) continue;
            seenInterp.insert(interpIdx);

            std::string nodeName = resolveString(cb.nodeNameIdx);
            if (!opt.noBlacklist && isBlacklistedName(nodeName, opt.extraBlacklist)) {
                skippedByBlacklist++;
                if (opt.verbose) log << "Skipping bone \"" << nodeName << "\" (interp#" << interpIdx << "): blacklisted name\n";
                continue;
            }

            ByteReader ir(raw.data() + blockOffset[interpIdx], header.blockSizes[interpIdx]);
            TransformInterpolator ti = TransformInterpolator::read(ir, header.blockSizes[interpIdx]);
            int dataIdx = ti.dataRef;
            if (dataIdx < 0 || dataIdx >= static_cast<int>(header.numBlocks)) continue;
            if (blockType(dataIdx) != "NiTransformData" && blockType(dataIdx) != "NiKeyframeData") continue;

            BoneJob job;
            job.interpBlockIdx = interpIdx;
            job.dataBlockIdx = dataIdx;
            job.interp = ti;
            job.name = nodeName;
            jobs.push_back(job);
        }
    }
    for (auto& job : jobs) {
        auto it = seqRangeOf.find(job.interpBlockIdx);
        if (it != seqRangeOf.end()) { job.seqT0 = it->second.first; job.seqT1 = it->second.second; job.haveSeqRange = true; }
    }
    // Converting a bone replaces its NiTransformData block with an empty
    // placeholder. If any OTHER NiTransformInterpolator that we are not
    // converting (blacklisted name, not a candidate, orphaned, ...) points at
    // the same data block, it would silently lose its animation. Exporters
    // give every interpolator its own block so this is not expected in
    // practice, but leaving such a bone as plain keys is the safe outcome.
    {
        std::map<int, int> referrers;    // data block -> NiTransformInterpolator count (all)
        std::map<int, int> jobReferrers; // data block -> jobs pointing at it
        for (uint32_t i = 0; i < header.numBlocks; ++i) {
            if (blockType(i) != "NiTransformInterpolator") continue;
            try {
                ByteReader ir(raw.data() + blockOffset[i], header.blockSizes[i]);
                TransformInterpolator ti = TransformInterpolator::read(ir, header.blockSizes[i]);
                ++referrers[ti.dataRef];
            } catch (const std::exception&) {
                // Unparseable interpolators are left alone (and never become jobs).
            }
        }
        for (auto& job : jobs) ++jobReferrers[job.dataBlockIdx];
        std::vector<BoneJob> unshared;
        unshared.reserve(jobs.size());
        for (auto& job : jobs) {
            if (referrers[job.dataBlockIdx] > jobReferrers[job.dataBlockIdx]) {
                if (opt.verbose) log << "Skipping bone \"" << job.name << "\" (interp#" << job.interpBlockIdx
                                     << "): its key data (block " << job.dataBlockIdx
                                     << ") is shared with an interpolator that is not being converted\n";
                continue;
            }
            unshared.push_back(std::move(job));
        }
        jobs = std::move(unshared);
    }
    result.bonesSkippedBlacklist = skippedByBlacklist;
    if (skippedByBlacklist && !opt.verbose) {
        log << "Skipped " << skippedByBlacklist << " blacklisted bone(s) "
               "(weapon/attach/IK names, or containing \"##\"); use -v to see which, "
               "or --no-blacklist to convert them anyway.\n";
    }

    if (jobs.empty()) {
        log << "No convertible NiTransformInterpolator/NiTransformData "
               "channels found (file may already use B-splines, or "
               "contains no NiControllerSequence).\n";
        if (!opt.dryRun) {
            // Copy the file through unchanged rather than writing nothing
            // at all -- in batch mode, every source file should have a
            // corresponding output file, even if there was nothing in it
            // to compress. Writing nothing here was a real gap: found
            // when a 14,313-file real batch produced only ~12,000 output
            // files, because every file with no convertible bones (some
            // camera/text-key-only sequences, files that already use
            // B-splines, etc.) simply vanished from the mirrored output
            // instead of being copied through.
            //
            // The in-place skip (inPath == outPath, nothing left to do)
            // only holds when `raw` still matches what's on disk -- with
            // --dedupe, `raw` may already differ from the on-disk original
            // even though no bone was B-spline-eligible, so that case must
            // still write.
            result.outputBytes = raw.size();
            result.bytes = std::move(raw);
            result.modified = result.dedupeBlocksTouched > 0;
            result.wroteFile = true;
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
    //
    // Each job's fit is independent of every other job's (read-only
    // access to jobData, writes only to that job's own ChannelFit
    // fields), so this loop is a safe target for intra-file
    // parallelism. In practice it isn't worth it: even a 64-bone file's
    // fits complete in a small fraction of a second single-threaded,
    // since each channel's linear solve is a tiny (tens-to-low-hundreds
    // unknowns) dense system -- thread launch/join overhead would
    // dominate. Parallelism pays off across *files* instead; see the
    // batch-mode thread pool in main().
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

    for (size_t j = 0; j < jobs.size(); ++j) {
        auto& job = jobs[j];
        ByteReader dr(fitBase + fitBlockOffset[job.dataBlockIdx], fitBlockSizes[job.dataBlockIdx]);
        jobData[j] = TransformData::read(dr, fitBlockSizes[job.dataBlockIdx]);
        origKeyBytes += header.blockSizes[job.dataBlockIdx];

        // Every channel on this bone is fitted over the SAME parametric range
        // (the sequence range, see resolveBoneRange): one start/stop per
        // interpolator.
        double boneT0, boneT1;
        resolveBoneRange(job, jobData[j], boneT0, boneT1);
        job.boneT0 = boneT0; job.boneT1 = boneT1;
        // A clip shorter than ~4 frames at 61 Hz has too few samples to fit
        // meaningfully: leave the bone as plain keys.
        if (!opt.noShortClipSkip && (boneT1 - boneT0) * 61.000004 < 4.0) {
            job.lossless = true;
            continue;
        }
        BoneSamples bs;
        if (!buildBoneSamples(jobData[j], opt.skip, boneT0, boneT1, !opt.noStaticChannels, bs)) {
            job.lossless = true; // NaN/Inf in the source keys
            continue;
        }
        for (int i = 0; i < 3; ++i) {
            job.staticCh[i] = bs.ch[i].constant;
            job.staticVal[i] = bs.ch[i].constValue;
        }
        const double duration = boneT1 - boneT0;
        const double budget[3] = {opt.maxTrans, opt.maxRotDeg, opt.maxScale};
        ChannelFit fits[3];
        int boneM = 0;

        bool budgetMode = opt.forcedControlPoints <= 0 && !opt.useTolerance;
        bool anyBudget = false;
        for (int i = 0; i < 3; ++i) anyBudget = anyBudget || (bs.ch[i].present && budget[i] > 0);

        if (budgetMode && anyBudget) {
            // Error-bounded: start at the ratio's M, keep the smallest M whose
            // quantised bytes pass the budget (see searchBoneM).
            size_t samples = 0;
            for (int i = 0; i < 3; ++i) samples = std::max(samples, bs.ch[i].keyCount);
            int sample61 = static_cast<int>(std::lround(duration * 61.000004)) + 1;
            int maxM = std::max(2 * static_cast<int>(samples), sample61);
            int m0 = ratioControlPoints(duration, opt.ratio, opt.maxControlPoints, opt.minControlPoints);
            if (opt.maxControlPoints > 0) maxM = std::min(maxM, opt.maxControlPoints);
            maxM = std::max(maxM, m0);
            if (!searchBoneM(bs, m0, maxM, opt.compact16, boneT0, boneT1, budget, fits, boneM)) {
                job.lossless = true;
                if (opt.verbose) log << "Keeping bone \"" << job.name << "\" (interp#" << job.interpBlockIdx
                                     << ") as plain keys: no M up to " << maxM << " met the error budget\n";
                continue;
            }
        } else {
            // Forced M, --tolerance (per-channel adaptive M, then the bone's
            // max), or ratio only (every budget disabled): no verification.
            if (opt.forcedControlPoints > 0) boneM = opt.forcedControlPoints;
            else if (!opt.useTolerance) boneM = ratioControlPoints(duration, opt.ratio, opt.maxControlPoints, opt.minControlPoints);
            else {
                boneM = bspline::kDegree + 1;
                for (int i = 0; i < 3; ++i) {
                    if (!bs.ch[i].present) continue;
                    int maxM = static_cast<int>(bs.ch[i].ts.size());
                    if (opt.maxControlPoints > 0) maxM = std::min(maxM, opt.maxControlPoints);
                    auto f = bspline::fitBSplineAdaptive(bs.ch[i].ts, bs.ch[i].vals, opt.tolerance, bspline::kDegree + 1, maxM);
                    boneM = std::max(boneM, static_cast<int>(f.controlPoints.size()));
                }
            }
            for (int i = 0; i < 3; ++i) fits[i] = fitChannel(bs.ch[i], boneM, opt.compact16, boneT0, boneT1);
        }
        job.translation = std::move(fits[0]);
        job.rotation = std::move(fits[1]);
        job.scale = std::move(fits[2]);
        // No spline channel at all (everything static or absent): the interpolator
        // still names a basis block, so point it at the smallest valid one.
        if (!bs.ch[0].present && !bs.ch[1].present && !bs.ch[2].present) boneM = bspline::kDegree + 1;
        jobM[j] = boneM;

        // Every present channel MUST come out with exactly boneM control
        // points: they all share one basis block (one M) per the file format,
        // so fewer would make appendChannel write fewer pool elements than the
        // basis block declares, a silent out-of-bounds read at playback. See
        // bspline.hpp's fitBSpline comment (M is never clamped to the sample
        // count).
        if ((job.translation.present && (int)job.translation.fit.controlPoints.size() != boneM) ||
            (job.rotation.present && (int)job.rotation.fit.controlPoints.size() != boneM) ||
            (job.scale.present && (int)job.scale.fit.controlPoints.size() != boneM)) {
            throw std::runtime_error(
                "Internal error: bone \"" + job.name + "\" (interp#" + std::to_string(job.interpBlockIdx) +
                ") has a channel whose fit didn't come back at the bone's shared M=" + std::to_string(boneM) +
                " -- refusing to write a file that would corrupt at playback.");
        }
    }

    if (opt.useTolerance && opt.maxControlPoints > 0) {
        bool anyOverTolerance = false;
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (jobM[j] < opt.maxControlPoints) continue;
            auto& job = jobs[j];
            if (job.translation.present && job.translation.fit.maxError > opt.tolerance) anyOverTolerance = true;
            if (job.rotation.present && job.rotation.fit.maxError > opt.tolerance) anyOverTolerance = true;
            if (job.scale.present && job.scale.fit.maxError > opt.tolerance) anyOverTolerance = true;
        }
        if (anyOverTolerance) {
            log << "Warning: hit --max-control-points (" << opt.maxControlPoints
                << ") before every channel met --tolerance " << opt.tolerance
                << " on at least one bone -- some channels are less accurate "
                   "than requested. This usually means an unusually long/dense "
                   "source animation; raise --max-control-points if you need "
                   "tighter accuracy, or set it to 0 (no cap, the default).\n";
        }
    }

    // ---- Drop bones where B-spline conversion at the shared M wouldn't
    //      actually shrink the file. Real Bethesda-compressed files
    //      consistently leave low-motion bones (typically 2-key channels)
    //      as plain NiTransformInterpolator rather than pay control-point
    //      overhead for zero benefit. We don't have the original tool's exact
    //      eligibility formula (it isn't in FalloutNV.exe), so this
    //      reconstructs its *effect* by
    //      computing the actual net file-size delta of converting this
    //      bone: converting replaces a 36-byte NiTransformInterpolator
    //      body with an 84-byte (60 if not --compact16)
    //      NiBSplineCompTransformInterpolator body, shrinks the original
    //      NiTransformData block to a 12-byte empty placeholder (see
    //      TransformData::writeEmpty), and adds this bone's control
    //      points to the shared pool -- NOT simply "pool bytes vs
    //      original data size" (an earlier version of this check compared
    //      those two numbers directly and got the sign wrong on the
    //      interpolator-growth and shrink-to-12 terms, occasionally
    //      making --compact16 output *bigger* than converting everything
    //      unconditionally would have -- caught by comparing real
    //      before/after file sizes, not just counts).
    // Can the unreferenced NiTransformData blocks be removed outright (see
    // remapBlockRefs)? Needs every block type in the file to be one we know
    // how to remap, and a footer that is exactly a root-reference list.
    bool dropPossible = !opt.keepPlaceholders;
    if (dropPossible) {
        for (uint32_t i = 0; i < header.numBlocks && dropPossible; ++i)
            if (!blockTypeRefsKnown(blockType(static_cast<int>(i)))) dropPossible = false;
        if (dropPossible) {
            uint32_t nRoots = 0;
            if (trailerLen < 4) dropPossible = false;
            else {
                std::memcpy(&nRoots, raw.data() + trailerStart, 4);
                if (trailerLen != 4 + 4 * static_cast<size_t>(nRoots)) dropPossible = false;
            }
        }
    }
    {
        size_t bytesPerComponent = opt.compact16 ? 2 : 4;
        size_t newInterpBodySize = opt.compact16 ? 84 : 60;
        constexpr size_t kOldInterpBodySize = 36; // NiQuatTransform(32) + dataRef(4)
        constexpr size_t kShrunkDataSize = 12;    // TransformData::writeEmpty()
        std::vector<BoneJob> keptJobs;
        std::vector<TransformData> keptJobData;
        std::vector<int> keptJobM;
        keptJobs.reserve(jobs.size());
        keptJobData.reserve(jobs.size());
        keptJobM.reserve(jobs.size());
        size_t skippedNotWorthwhile = 0;
        for (size_t j = 0; j < jobs.size(); ++j) {
            auto& job = jobs[j];
            int m = jobM[j];
            if (job.lossless) {
                result.bonesKeptLossless++;
                continue;
            }
            if (!jobControlPointsFinite(job)) {
                if (opt.verbose) {
                    log << "Skipping bone \"" << job.name << "\" (interp#" << job.interpBlockIdx
                        << "): the fit produced a non-finite control point\n";
                }
                continue;
            }
            size_t poolBytes = 0;
            if (job.translation.present) poolBytes += static_cast<size_t>(m) * 3 * bytesPerComponent;
            if (job.rotation.present) poolBytes += static_cast<size_t>(m) * 4 * bytesPerComponent;
            if (job.scale.present) poolBytes += static_cast<size_t>(m) * 1 * bytesPerComponent;
            size_t originalDataBytes = header.blockSizes[job.dataBlockIdx];
            // Net change to the file if this bone IS converted (positive = grows).
            long long netDelta = static_cast<long long>(newInterpBodySize) - static_cast<long long>(kOldInterpBodySize)
                                + static_cast<long long>(dropPossible ? 0 : kShrunkDataSize) -static_cast<long long>(originalDataBytes)
                                + static_cast<long long>(poolBytes);
            if (!opt.forceConvertAll && netDelta >= 0) {
                skippedNotWorthwhile++;
                if (opt.verbose) {
                    log << "Skipping bone \"" << job.name << "\" (interp#" << job.interpBlockIdx
                        << "): converting at M=" << m << " would grow the file by "
                        << netDelta << " byte(s) -- not worthwhile\n";
                }
                continue;
            }
            keptJobs.push_back(std::move(job));
            keptJobData.push_back(std::move(jobData[j]));
            keptJobM.push_back(m);
        }
        result.bonesSkippedNotWorthwhile = skippedNotWorthwhile;
        if (skippedNotWorthwhile && !opt.verbose) {
            log << "Skipped " << skippedNotWorthwhile << " bone(s) where B-spline conversion "
                   "wouldn't save space (use -v to see which, or --force-convert-all to convert anyway).\n";
        }
        jobs = std::move(keptJobs);
        jobData = std::move(keptJobData);
        jobM = std::move(keptJobM);
        origKeyBytes = 0;
        for (auto& job : jobs) origKeyBytes += header.blockSizes[job.dataBlockIdx];

        if (jobs.empty()) {
            log << "No bone remained eligible after the size-based conversion check "
                   "(every candidate would have gotten bigger, not smaller, or had an unusable fit).\n";
            if (!opt.dryRun) {
                // See the identical note on the other early-return above:
                // with --dedupe, `raw` may differ from disk even here.
                result.outputBytes = raw.size();
                result.bytes = std::move(raw);
                result.modified = result.dedupeBlocksTouched > 0;
                result.wroteFile = true;
            }
            return result;
        }
    }

    if (opt.verbose) {
        std::set<int> distinctM(jobM.begin(), jobM.end());
        log << "Control-point count (M) per bone -- " << distinctM.size()
            << " distinct value(s) in use: ";
        for (int m : distinctM) log << m << " ";
        log << "\n";
        for (size_t j = 0; j < jobs.size(); ++j) {
            auto& job = jobs[j];
            log << "bone interp#" << job.interpBlockIdx
                << (job.name.empty() ? "" : " \"" + job.name + "\"") << " M=" << jobM[j] << ": ";
            if (job.translation.present) log << "trans err=" << job.translation.fit.maxError << "  ";
            if (job.rotation.present) log << "rot err=" << job.rotation.fit.maxError << "  ";
            if (job.scale.present) log << "scale err=" << job.scale.fit.maxError;
            log << "\n";
        }
    }

    result.bonesConverted = jobs.size();

    if (opt.dryRun) {
        log << "Dry run: " << jobs.size() << " bone channel(s) would be converted.\n";
        return result;
    }

    // ---- Assemble shared control-point pool + one basis block per distinct M ----
    // The control-point pool is still one shared NiBSplineData block (that
    // part was never file-wide-M-dependent -- it's just concatenated
    // per-channel arrays addressed by handle+dimension*M). Basis data is
    // different: real vanilla files group bones into a handful of
    // NiBSplineBasisData blocks by matching M value, not one block for the
    // whole file (see the jobM comment above) -- each bone's basisDataRef
    // points at whichever group its own M belongs to.
    std::vector<float> floatPool;
    std::vector<int16_t> compactPool;
    std::vector<std::vector<uint8_t>> newInterpBytes(jobs.size());

    int nextNewBlockIndex = static_cast<int>(header.numBlocks);
    int sharedSplineDataFinalIndex = nextNewBlockIndex++;

    // Assign each distinct M value seen a basis-block index, in first-seen
    // order (order doesn't matter functionally, just needs to be stable).
    std::map<int, int> basisIndexForM; // M -> block index
    for (int m : jobM) {
        if (basisIndexForM.find(m) == basisIndexForM.end())
            basisIndexForM[m] = nextNewBlockIndex++;
    }

    for (size_t j = 0; j < jobs.size(); ++j) {
        auto& job = jobs[j];
        BSplineTransformInterpolatorOut out;
        out.compact = opt.compact16;
        out.transform = job.interp.transform; // static fallback pose for any missing channel
        float t0 = 1e30f, t1 = -1e30f;
        if (job.translation.present) { t0 = std::min(t0, job.translation.startTime); t1 = std::max(t1, job.translation.stopTime); }
        if (job.rotation.present)    { t0 = std::min(t0, job.rotation.startTime);    t1 = std::max(t1, job.rotation.stopTime); }
        if (job.scale.present)       { t0 = std::min(t0, job.scale.startTime);       t1 = std::max(t1, job.scale.stopTime); }
        if (t0 > t1) {
            // No spline channel (all static/absent): span the bone's own range.
            if (job.boneT1 > job.boneT0) { t0 = static_cast<float>(job.boneT0); t1 = static_cast<float>(job.boneT1); }
            else { t0 = 0; t1 = 1; }
        }
        out.startTime = t0; out.stopTime = t1;
        // Constant channels live in the static pose (their handle stays "no data").
        if (job.staticCh[0]) {
            out.transform.translation.x = static_cast<float>(job.staticVal[0][0]);
            out.transform.translation.y = static_cast<float>(job.staticVal[0][1]);
            out.transform.translation.z = static_cast<float>(job.staticVal[0][2]);
        }
        if (job.staticCh[1]) { // quaternion order w, x, y, z
            out.transform.rotation.w = static_cast<float>(job.staticVal[1][0]);
            out.transform.rotation.x = static_cast<float>(job.staticVal[1][1]);
            out.transform.rotation.y = static_cast<float>(job.staticVal[1][2]);
            out.transform.rotation.z = static_cast<float>(job.staticVal[1][3]);
        }
        if (job.staticCh[2]) out.transform.scale = static_cast<float>(job.staticVal[2][0]);

        out.translationHandle = appendChannel(job.translation, opt.compact16, floatPool, compactPool, out.translationOffset, out.translationHalfRange);
        out.rotationHandle    = appendChannel(job.rotation,    opt.compact16, floatPool, compactPool, out.rotationOffset,    out.rotationHalfRange);
        out.scaleHandle       = appendChannel(job.scale,       opt.compact16, floatPool, compactPool, out.scaleOffset,       out.scaleHalfRange);

        out.splineDataRef = sharedSplineDataFinalIndex;
        out.basisDataRef = basisIndexForM[jobM[j]];

        newInterpBytes[j] = out.serialize();
    }

    BSplineData sharedData;
    sharedData.floatControlPoints = std::move(floatPool);
    sharedData.compactControlPoints = std::move(compactPool);
    auto sharedDataBytes = sharedData.serialize();
    const size_t sharedDataSize = sharedDataBytes.size();

    // ---- Rebuild header block tables ----
    std::string interpTypeName = opt.compact16 ? "NiBSplineCompTransformInterpolator" : "NiBSplineTransformInterpolator";
    int interpTypeIdx = header.findOrAddBlockType(interpTypeName);
    int basisTypeIdx = header.findOrAddBlockType("NiBSplineBasisData");
    int splineDataTypeIdx = header.findOrAddBlockType("NiBSplineData");

    uint32_t newNumBlocks = header.numBlocks + 1 /* shared NiBSplineData */
                           + static_cast<uint32_t>(basisIndexForM.size()) /* one NiBSplineBasisData per distinct M */;
    std::vector<uint16_t> newBlockTypeIndex = header.blockTypeIndex;
    std::vector<uint32_t> newBlockSizes = header.blockSizes;
    newBlockTypeIndex.resize(newNumBlocks);
    newBlockSizes.resize(newNumBlocks);

    // Track, per original block index, replacement bytes (for modified
    // interpolators and the new blocks); everything else copies as-is.
    std::map<int, std::vector<uint8_t>> replacedBlockBytes;
    for (size_t j = 0; j < jobs.size(); ++j) {
        auto& job = jobs[j];
        newBlockTypeIndex[job.interpBlockIdx] = static_cast<uint16_t>(interpTypeIdx);
        replacedBlockBytes[job.interpBlockIdx] = std::move(newInterpBytes[j]);
    }
    newBlockTypeIndex[sharedSplineDataFinalIndex] = static_cast<uint16_t>(splineDataTypeIdx);
    replacedBlockBytes[sharedSplineDataFinalIndex] = std::move(sharedDataBytes);

    for (auto& [m, blockIdx] : basisIndexForM) {
        BSplineBasisData basis;
        basis.numControlPoints = static_cast<uint32_t>(m);
        newBlockTypeIndex[blockIdx] = static_cast<uint16_t>(basisTypeIdx);
        replacedBlockBytes[blockIdx] = basis.serialize();
    }

    // ---- Serialize the output file ----
    // Two layouts. Placeholder mode keeps the block count and index space:
    // each converted bone's old NiTransformData becomes a 12-byte empty
    // block. Drop mode removes those blocks entirely, which means compacting
    // the index space and remapping every reference (see remapBlockRefs);
    // it returns false -- and we fall back to placeholders -- if any block
    // can't be remapped with certainty.
    auto buildOutput = [&](bool drop, ByteWriter& out) -> bool {
        std::vector<uint8_t> removed(newNumBlocks, 0);
        if (drop) for (auto& job : jobs) removed[job.dataBlockIdx] = 1;
        std::vector<int32_t> newIndex(newNumBlocks, -1);
        int32_t kept = 0;
        for (uint32_t i = 0; i < newNumBlocks; ++i) if (!removed[i]) newIndex[i] = kept++;

        std::set<int> placeholderIdx;
        if (!drop) for (auto& job : jobs) placeholderIdx.insert(job.dataBlockIdx);

        std::vector<std::vector<uint8_t>> blocks;
        std::vector<uint16_t> types;
        for (uint32_t i = 0; i < newNumBlocks; ++i) {
            if (removed[i]) continue;
            std::vector<uint8_t> bytes;
            auto it = replacedBlockBytes.find(static_cast<int>(i));
            if (it != replacedBlockBytes.end()) bytes = it->second;
            else if (placeholderIdx.count(static_cast<int>(i))) bytes = TransformData::writeEmpty();
            else bytes.assign(raw.data() + blockOffset[i], raw.data() + blockOffset[i] + header.blockSizes[i]);
            uint16_t t = newBlockTypeIndex[i];
            if (drop && !remapBlockRefs(header.blockTypes[t], bytes, newIndex, bsVersion)) return false;
            blocks.push_back(std::move(bytes));
            types.push_back(t);
        }

        std::vector<uint8_t> footer(raw.data() + trailerStart, raw.data() + trailerStart + trailerLen);
        if (drop) {
            uint32_t nRoots;
            std::memcpy(&nRoots, footer.data(), 4);
            for (uint32_t k = 0; k < nRoots; ++k)
                if (!patchRef(footer, 4 + 4 * static_cast<size_t>(k), newIndex)) return false;
        }

        Header oh = header;
        if (drop) {
            // Block types nothing uses any more (typically NiTransformData
            // when every bone was converted) are dropped from the type table.
            std::vector<int> typeMap(header.blockTypes.size(), -1);
            std::vector<std::string> usedTypes;
            for (auto& t : types) {
                if (typeMap[t] < 0) { typeMap[t] = static_cast<int>(usedTypes.size()); usedTypes.push_back(header.blockTypes[t]); }
                t = static_cast<uint16_t>(typeMap[t]);
            }
            oh.blockTypes = usedTypes;
            oh.numBlockTypes = static_cast<uint16_t>(usedTypes.size());
        }
        oh.numBlocks = static_cast<uint32_t>(blocks.size());
        oh.blockTypeIndex = types;
        oh.blockSizes.resize(blocks.size());
        for (size_t i = 0; i < blocks.size(); ++i) oh.blockSizes[i] = static_cast<uint32_t>(blocks[i].size());

        writeHeader(out, oh);
        for (auto& bl : blocks) out.raw(bl);
        if (!footer.empty()) out.raw(footer);
        return true;
    };

    ByteWriter out;
    if (!(dropPossible && buildOutput(true, out))) {
        out = ByteWriter();
        buildOutput(false, out);
    }

    // The per-bone size check above only weighs each bone's own blocks and
    // pool bytes. Converting also costs the file a fixed overhead it can't
    // see: the shared NiBSplineData and per-M basis blocks, their block-type
    // and block-size table entries, and up to three new type-name strings
    // (~100 bytes). On small files a handful of bones that each save a few
    // bytes can lose to that, so the "compressed" file came out bigger (146
    // of a 12k-file mod batch) -- and lossy for nothing. Measuring the real
    // serialised size can't be wrong where an up-front estimate could be, so
    // compare here and fall back to the same output as "no bone eligible".
    // Against `raw`, i.e. after dedupe, so its lossless savings are kept.
    // --force-convert-all opts out of size checks, so it opts out of this too.
    if (!opt.forceConvertAll && out.buf.size() >= raw.size()) {
        log << "B-spline output (" << out.buf.size() << " bytes) is not smaller than the "
            << (result.dedupeBlocksTouched > 0 ? "deduped " : "") << "original (" << raw.size()
            << " bytes); keeping the original keys.\n";
        result.bonesConverted = 0;
        result.outputBytes = raw.size();
        result.bytes = std::move(raw);
        result.modified = result.dedupeBlocksTouched > 0;
        result.wroteFile = true;
        return result;
    }

    result.bytes = std::move(out.buf);
    result.modified = true;
    result.wroteFile = true;
    result.outputBytes = result.bytes.size();

    size_t newControlBytes = sharedDataSize;
    for (auto& [m, blockIdx] : basisIndexForM) newControlBytes += replacedBlockBytes[blockIdx].size();
    for (size_t j = 0; j < jobs.size(); ++j) newControlBytes += replacedBlockBytes[jobs[j].interpBlockIdx].size();

    log << "Converted " << jobs.size() << " bone channel(s).\n";
    if (result.dedupeBlocksTouched > 0) {
        log << "  Dedupe pre-pass: removed " << result.dedupeKeysRemoved << " key(s) across "
            << result.dedupeBlocksTouched << " block(s)\n";
    }
    log << "  Original key data:      " << origKeyBytes << " bytes\n"
        << "  New spline/control data: " << newControlBytes << " bytes\n"
        // result.inputBytes is the true on-disk original size (captured
        // before the optional dedupe pre-pass may have shrunk `raw`), so
        // this reflects the combined effect of both steps when --dedupe
        // is on, not just the B-spline stage's own delta.
        << "  File size: " << result.inputBytes << " -> " << result.outputBytes << " bytes ("
        << (100.0 * (double)result.outputBytes / (double)result.inputBytes) << "%)\n";
    return result;
}

} // namespace kfc

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
// dedupe_types.hpp - full-fidelity NiTransformData parsing + "remove excess
// repeated keys" reduction, ported from S'Lanter's NIF Helper's actual
// "Optimize Animations" feature (TProcOptimizeKF.ProcessFile /
// Optimize() in ProcOptimizeKF.pas, from the TES5Edit-dev-4.1.6 source
// tree under Sniff/Proc/).
//
// Unlike kfcompress's TransformData (niftypes.hpp), which intentionally
// DISCARDS tangent/TBC data because it refits a fresh curve through key
// *values* only, this header preserves every field of every key exactly so
// non-reduced key groups round-trip byte-for-byte and reduced ones stay
// lossless.
//
// The reduction algorithm itself, ported from the Pascal:
//   - Walks each key array backward, comparing three consecutive keys'
//     "value" at a time (cached locals, mirroring the original -- see
//     optimizeRun below).
//   - If a key's value equals both its immediate neighbors', it is deleted.
//   - Repeating this leftward across the array collapses any run of N
//     identical consecutive values down to just its first and last key --
//     lossless under linear interpolation, since every point strictly
//     between two identical values interpolates to that same value
//     regardless of how many redundant samples originally existed there.
//   - The first and last key of the array are never deleted.
//
// Divergence from the Pascal original (deliberate, documented in README):
// for QUADRATIC_KEY/TBC_KEY groups, the original's KeyValue() compares
// ONLY the tangent/TBC fields and ignores the key's actual value -- this
// port instead compares value AND tangents together for those interpolation
// types, so a reduction there is always strictly lossless rather than
// matching that apparent upstream shortcut. LINEAR_KEY groups (which have
// no tangent data at all -- the overwhelming majority of "repeated key"
// runs in real assets) behave identically to the original.
#pragma once
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "niftypes.hpp"

namespace kfdedupe {

// ByteReader/ByteWriter live in the global namespace (byteio.hpp), not
// inside nif:: -- only bring in the nif:: types that actually live there.
using nif::KeyType;
using nif::Quaternion;
using nif::Vector3;

using nif::readVector3;
using nif::readQuaternion;
inline void writeVec3(ByteWriter& w, const Vector3& v) { w.f32(v.x); w.f32(v.y); w.f32(v.z); }
inline void writeQuat(ByteWriter& w, const Quaternion& q) { w.f32(q.w); w.f32(q.x); w.f32(q.y); w.f32(q.z); }

// epsilon == 0 is exact bit equality (-0.0f == 0.0f is already true in IEEE
// 754); epsilon > 0 treats values within that absolute tolerance as equal.
// Passing a nonzero epsilon makes the reduction bounded-error instead of
// strictly lossless -- see the DedupePassResult/dedupeFileBytes comment and
// README for why you'd want that (matching S'Lanter's NIF Helper's own
// string-formatted, not bit-exact, comparison).
inline bool eqf(float a, float b, float epsilon = 0.0f) {
    return epsilon > 0.0f ? std::fabs(a - b) <= epsilon : a == b;
}
inline bool eqv3(const Vector3& a, const Vector3& b, float epsilon = 0.0f) {
    return eqf(a.x, b.x, epsilon) && eqf(a.y, b.y, epsilon) && eqf(a.z, b.z, epsilon);
}
inline bool eqq(const Quaternion& a, const Quaternion& b, float epsilon = 0.0f) {
    return eqf(a.w, b.w, epsilon) && eqf(a.x, b.x, epsilon) && eqf(a.y, b.y, epsilon) && eqf(a.z, b.z, epsilon);
}

// ---------------------------------------------------------------- key types
// Every field any key type in the format can carry, always populated on
// read regardless of which are actually meaningful for this group's
// interpolation type -- so write() can always re-emit exactly what applies.

struct FFloatKey {
    float time = 0, value = 0;
    float tanFwd = 0, tanBack = 0; // QUADRATIC_KEY
    float t = 0, b = 0, c = 0;     // TBC_KEY
};

struct FVec3Key {
    float time = 0;
    Vector3 value;
    Vector3 tanFwd, tanBack; // QUADRATIC_KEY
    float t = 0, b = 0, c = 0; // TBC_KEY
};

struct FQuatKey {
    float time = 0;
    Quaternion value;
    float t = 0, b = 0, c = 0; // TBC_KEY (QuatKey has no quadratic tangent variant)
};

// value+tangent equality, gated on which fields actually apply for `interp`.
inline bool sameKey(const FFloatKey& a, const FFloatKey& b, KeyType interp, float epsilon = 0.0f) {
    if (!eqf(a.value, b.value, epsilon)) return false;
    if (interp == KeyType::QUADRATIC_KEY) return eqf(a.tanFwd, b.tanFwd, epsilon) && eqf(a.tanBack, b.tanBack, epsilon);
    if (interp == KeyType::TBC_KEY) return eqf(a.t, b.t, epsilon) && eqf(a.b, b.b, epsilon) && eqf(a.c, b.c, epsilon);
    return true;
}
inline bool sameKey(const FVec3Key& a, const FVec3Key& b, KeyType interp, float epsilon = 0.0f) {
    if (!eqv3(a.value, b.value, epsilon)) return false;
    if (interp == KeyType::QUADRATIC_KEY) return eqv3(a.tanFwd, b.tanFwd, epsilon) && eqv3(a.tanBack, b.tanBack, epsilon);
    if (interp == KeyType::TBC_KEY) return eqf(a.t, b.t, epsilon) && eqf(a.b, b.b, epsilon) && eqf(a.c, b.c, epsilon);
    return true;
}
inline bool sameKey(const FQuatKey& a, const FQuatKey& b, KeyType interp, float epsilon = 0.0f) {
    if (!eqq(a.value, b.value, epsilon)) return false;
    if (interp == KeyType::TBC_KEY) return eqf(a.t, b.t, epsilon) && eqf(a.b, b.b, epsilon) && eqf(a.c, b.c, epsilon);
    return true;
}

// ---------------------------------------------------------------- the
// reduction algorithm itself, a direct port of ProcOptimizeKF.pas's
// Optimize(): backward scan, three cached values at a time, delete the
// middle whenever it matches both neighbors. Returns the number of keys
// removed. Marks removed original indices rather than physically erasing
// mid-scan (erasing mid-scan is what the Pascal does, via a live TdfElement
// list) -- equivalent here because, exactly as in the original, a later
// deletion at index j never disturbs any index < j, and every `prev` this
// scan reads is always at an index smaller than any index deleted so far.
template <typename KeyT>
size_t optimizeRun(std::vector<KeyT>& keys, KeyType interp, float epsilon = 0.0f) {
    size_t n = keys.size();
    if (n < 3) return 0;

    std::vector<bool> removed(n, false);
    bool changed = false;

    KeyT nextK = keys[n - 1];
    KeyT curK = keys[n - 2];
    for (size_t j = n - 2; j >= 1; --j) {
        KeyT prevK = keys[j - 1];
        if (sameKey(curK, prevK, interp, epsilon) && sameKey(curK, nextK, interp, epsilon)) {
            removed[j] = true;
            changed = true;
        }
        nextK = curK;
        curK = prevK;
        if (j == 1) break; // avoid size_t underflow on the final decrement
    }

    if (!changed) return 0;
    std::vector<KeyT> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) if (!removed[i]) out.push_back(keys[i]);
    size_t numRemoved = n - out.size();
    keys = std::move(out);
    return numRemoved;
}

// ---------------------------------------------------------------- key groups

struct FFloatGroup {
    KeyType interp = KeyType::LINEAR_KEY;
    std::vector<FFloatKey> keys;
    bool present() const { return !keys.empty(); }

    static FFloatGroup read(ByteReader& r) {
        FFloatGroup g;
        uint32_t n = r.u32();
        if (n == 0) return g;
        g.interp = nif::checkKeys(r, n, r.u32(), 4);
        g.keys.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            FFloatKey k;
            k.time = r.f32();
            k.value = r.f32();
            if (g.interp == KeyType::QUADRATIC_KEY) { k.tanFwd = r.f32(); k.tanBack = r.f32(); }
            else if (g.interp == KeyType::TBC_KEY) { k.t = r.f32(); k.b = r.f32(); k.c = r.f32(); }
            g.keys[i] = k;
        }
        return g;
    }
    void write(ByteWriter& w) const {
        w.u32(static_cast<uint32_t>(keys.size()));
        if (keys.empty()) return;
        w.u32(static_cast<uint32_t>(interp));
        for (auto& k : keys) {
            w.f32(k.time); w.f32(k.value);
            if (interp == KeyType::QUADRATIC_KEY) { w.f32(k.tanFwd); w.f32(k.tanBack); }
            else if (interp == KeyType::TBC_KEY) { w.f32(k.t); w.f32(k.b); w.f32(k.c); }
        }
    }
    size_t optimize(float epsilon = 0.0f) { return optimizeRun(keys, interp, epsilon); }
};

struct FVec3Group {
    KeyType interp = KeyType::LINEAR_KEY;
    std::vector<FVec3Key> keys;
    bool present() const { return !keys.empty(); }

    static FVec3Group read(ByteReader& r) {
        FVec3Group g;
        uint32_t n = r.u32();
        if (n == 0) return g;
        g.interp = nif::checkKeys(r, n, r.u32(), 12);
        g.keys.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            FVec3Key k;
            k.time = r.f32();
            k.value = readVector3(r);
            if (g.interp == KeyType::QUADRATIC_KEY) { k.tanFwd = readVector3(r); k.tanBack = readVector3(r); }
            else if (g.interp == KeyType::TBC_KEY) { k.t = r.f32(); k.b = r.f32(); k.c = r.f32(); }
            g.keys[i] = k;
        }
        return g;
    }
    void write(ByteWriter& w) const {
        w.u32(static_cast<uint32_t>(keys.size()));
        if (keys.empty()) return;
        w.u32(static_cast<uint32_t>(interp));
        for (auto& k : keys) {
            w.f32(k.time); writeVec3(w, k.value);
            if (interp == KeyType::QUADRATIC_KEY) { writeVec3(w, k.tanFwd); writeVec3(w, k.tanBack); }
            else if (interp == KeyType::TBC_KEY) { w.f32(k.t); w.f32(k.b); w.f32(k.c); }
        }
    }
    size_t optimize(float epsilon = 0.0f) { return optimizeRun(keys, interp, epsilon); }
};

// ---------------------------------------------------------------- NiTransformData
// Mirrors nif::TransformData's layout exactly (see niftypes.hpp for the
// authoritative field-order comment) but keeps every byte instead of
// discarding tangents, so it can be re-serialized losslessly.
struct FTransformData {
    uint32_t numRotationKeys = 0;
    KeyType rotationType = KeyType::LINEAR_KEY;
    std::vector<FQuatKey> quatKeys;   // used if rotationType != XYZ
    FFloatGroup xyzRotations[3];      // used if rotationType == XYZ
    FVec3Group translations;
    FFloatGroup scales;

    bool hasRotation() const { return numRotationKeys != 0; }
    bool isXYZRotation() const { return hasRotation() && rotationType == KeyType::XYZ_ROTATION_KEY; }

    static FTransformData read(ByteReader& r, uint32_t blockSizeForValidation) {
        size_t startPos = r.pos;
        FTransformData d;
        d.numRotationKeys = r.u32();
        if (d.numRotationKeys != 0) {
            uint32_t rawType = r.u32();
            if (rawType != static_cast<uint32_t>(KeyType::XYZ_ROTATION_KEY))
                d.rotationType = nif::checkKeys(r, d.numRotationKeys, rawType, 16, false);
            else d.rotationType = KeyType::XYZ_ROTATION_KEY;
            if (d.rotationType != KeyType::XYZ_ROTATION_KEY) {
                d.quatKeys.resize(d.numRotationKeys);
                for (uint32_t i = 0; i < d.numRotationKeys; ++i) {
                    FQuatKey k;
                    k.time = r.f32();
                    k.value = readQuaternion(r);
                    if (d.rotationType == KeyType::TBC_KEY) { k.t = r.f32(); k.b = r.f32(); k.c = r.f32(); }
                    d.quatKeys[i] = k;
                }
            } else {
                for (int axis = 0; axis < 3; ++axis) d.xyzRotations[axis] = FFloatGroup::read(r);
            }
        }
        d.translations = FVec3Group::read(r);
        d.scales = FFloatGroup::read(r);

        size_t consumed = r.pos - startPos;
        if (consumed != blockSizeForValidation) {
            throw FormatError(
                "NiTransformData: parsed " + std::to_string(consumed) +
                " bytes but block size says " + std::to_string(blockSizeForValidation) +
                " -- this file's NiTransformData layout doesn't match what this tool "
                "supports (version 20.2.0.7 Bethesda). Aborting rather than risk "
                "silent corruption.");
        }
        return d;
    }

    void write(ByteWriter& w) const {
        if (!hasRotation()) {
            w.u32(0);
        } else if (!isXYZRotation()) {
            w.u32(static_cast<uint32_t>(quatKeys.size()));
            w.u32(static_cast<uint32_t>(rotationType));
            for (auto& k : quatKeys) {
                w.f32(k.time);
                writeQuat(w, k.value);
                if (rotationType == KeyType::TBC_KEY) { w.f32(k.t); w.f32(k.b); w.f32(k.c); }
            }
        } else {
            // numRotationKeys is just a nonzero "rotation present" flag in
            // XYZ mode, independent of any axis's own key count (each axis
            // group carries its own count prefix) -- preserved verbatim
            // rather than recomputed.
            w.u32(numRotationKeys);
            w.u32(static_cast<uint32_t>(rotationType));
            for (int axis = 0; axis < 3; ++axis) xyzRotations[axis].write(w);
        }
        translations.write(w);
        scales.write(w);
    }

    // Runs the reduction on every present key group. Returns total keys
    // removed across all channels (0 = nothing changed, block need not be
    // rewritten). epsilon == 0 (default) is the strictly lossless port of
    // ProcOptimizeKF.pas's Optimize(); epsilon > 0 treats keys within that
    // absolute per-component tolerance as equal too, which is no longer
    // provably lossless (a removed key's true value can drift from its
    // neighbors by up to ~2x epsilon once interpolated) but matches how
    // much more NIF Helper itself collapses in practice, since its own
    // comparison goes through a string-formatted value rather than bit-exact
    // float equality.
    size_t optimize(float epsilon = 0.0f) {
        size_t removed = 0;
        if (translations.present()) removed += translations.optimize(epsilon);
        if (scales.present()) removed += scales.optimize(epsilon);
        if (hasRotation()) {
            if (!isXYZRotation()) {
                size_t r = optimizeRun(quatKeys, rotationType, epsilon);
                removed += r;
                if (r) numRotationKeys = static_cast<uint32_t>(quatKeys.size());
            } else {
                for (int axis = 0; axis < 3; ++axis)
                    if (xyzRotations[axis].present()) removed += xyzRotations[axis].optimize(epsilon);
            }
        }
        return removed;
    }
};

// ---------------------------------------------------------------- whole-file pass
//
// Runs the reduction across every NiTransformData/NiKeyframeData block in a
// whole in-memory NIF file and returns the rebuilt bytes -- factored out so
// both the standalone kfdedupe CLI and kfcompress's optional --dedupe
// pre-pass share exactly one implementation instead of two copies that
// could drift. No graph walk, no blacklist: this is a lossless reduction,
// not a lossy approximation, so unlike kfcompress's B-spline conversion
// there's no bone that needs protecting from it -- every eligible block in
// the file is processed.
struct DedupePassResult {
    std::vector<uint8_t> bytes;  // rebuilt file bytes; only meaningful if changed
    size_t blocksTouched = 0;
    size_t keysRemoved = 0;
    bool changed = false;
    std::vector<std::string> verboseLines; // one per touched block, for -v reporting
};

inline DedupePassResult dedupeFileBytes(const std::vector<uint8_t>& raw, float epsilon = 0.0f) {
    DedupePassResult result;

    ByteReader hr(raw);
    nif::Header header = nif::readHeader(hr);

    // Same out-of-range-block-type-index hardening used throughout this
    // project -- a malformed/truncated file must fail cleanly, not crash.
    for (uint32_t i = 0; i < header.numBlocks; ++i) {
        if (header.blockTypeIndex[i] >= header.blockTypes.size()) {
            throw FormatError(
                "Block " + std::to_string(i) + " has an out-of-range block-type index "
                "(" + std::to_string(header.blockTypeIndex[i]) + " >= " +
                std::to_string(header.blockTypes.size()) + " known types) -- file is "
                "corrupt, truncated, or not actually this NIF version despite its extension.");
        }
    }

    std::vector<size_t> blockOffset(header.numBlocks);
    size_t cursor = hr.pos;
    for (uint32_t i = 0; i < header.numBlocks; ++i) {
        blockOffset[i] = cursor;
        cursor += header.blockSizes[i];
        if (cursor > raw.size()) {
            throw FormatError(
                "Block " + std::to_string(i) + "'s declared size runs past the end of "
                "the file (file is truncated or the block-size table is corrupt).");
        }
    }
    size_t trailerStart = cursor, trailerLen = raw.size() - cursor;

    auto blockType = [&](int idx) -> const std::string& { return header.blockTypes[header.blockTypeIndex[idx]]; };

    std::map<int, std::vector<uint8_t>> replacedBlockBytes;
    std::vector<uint32_t> newBlockSizes = header.blockSizes;

    for (uint32_t i = 0; i < header.numBlocks; ++i) {
        const std::string& ty = blockType(i);
        if (ty != "NiTransformData" && ty != "NiKeyframeData") continue;

        ByteReader dr(raw.data() + blockOffset[i], header.blockSizes[i]);
        FTransformData data = FTransformData::read(dr, header.blockSizes[i]);

        size_t removed = data.optimize(epsilon);
        if (removed == 0) continue;

        ByteWriter bw;
        data.write(bw);
        // A dedupe-only rewrite can only ever shrink or match the original
        // block size (keys are removed, never added or resized) -- if this
        // ever produced something larger or the round-trip byte count is
        // off, that's an internal bug, not something to write out silently.
        if (bw.buf.size() >= header.blockSizes[i]) {
            throw std::runtime_error(
                "Internal error: deduped NiTransformData block " + std::to_string(i) +
                " came out " + std::to_string(bw.buf.size()) + " bytes, not smaller than "
                "the original " + std::to_string(header.blockSizes[i]) + " -- refusing to "
                "write a suspicious result.");
        }

        result.blocksTouched++;
        result.keysRemoved += removed;
        result.verboseLines.push_back(
            "block#" + std::to_string(i) + " (" + ty + "): removed " + std::to_string(removed) +
            " key(s), " + std::to_string(header.blockSizes[i]) + " -> " + std::to_string(bw.buf.size()) + " bytes");

        replacedBlockBytes[i] = std::move(bw.buf);
        newBlockSizes[i] = static_cast<uint32_t>(replacedBlockBytes[i].size());
    }

    if (result.blocksTouched == 0) return result; // changed stays false

    // ---- Rebuild: same block count/types/order as the input, just
    //      smaller sizes for the blocks that got deduped. No index
    //      renumbering needed anywhere in the file. ----
    header.blockSizes = newBlockSizes;

    ByteWriter out;
    nif::writeHeader(out, header);
    for (uint32_t i = 0; i < header.numBlocks; ++i) {
        auto it = replacedBlockBytes.find(static_cast<int>(i));
        if (it != replacedBlockBytes.end()) out.raw(it->second);
        else out.raw(raw.data() + blockOffset[i], header.blockSizes[i]);
    }
    if (trailerLen) out.raw(raw.data() + trailerStart, trailerLen);

    result.bytes = std::move(out.buf);
    result.changed = true;
    return result;
}

} // namespace kfdedupe

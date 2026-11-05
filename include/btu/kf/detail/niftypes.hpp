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
// niftypes.hpp - NIF container parsing scoped to version 20.2.0.7 (Bethesda
// Fallout 3 / Fallout New Vegas / late-patch Oblivion), which is the only
// version this compressor supports. Field layouts are taken directly from
// the authoritative niftools/nifxml format spec (nif.xml), resolved against
// this fixed version so no ambiguity remains at runtime except for the
// handful of fields that are genuinely conditioned on the file's own
// BSStreamHeader version, which we read from the file itself.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "byteio.hpp"

namespace nif {

constexpr uint32_t kSupportedVersion = 0x14020007; // 20.2.0.7

struct Vector3 { float x = 0, y = 0, z = 0; };
struct Quaternion { float w = 1, x = 0, y = 0, z = 0; };

// "No data" sentinel for Translation/Rotation/Scale Handle fields. Despite
// the field being 4 bytes wide, real Bethesda-exported files use 0xFFFF
// (USHRT_MAX) here, matching the format spec's documented default -- NOT
// 0xFFFFFFFF. Confirmed against real Fallout NV .kf files: writing
// 0xFFFFFFFF here instead would make the engine read control point index
// 4294967295, an out-of-bounds read/crash.
constexpr uint32_t kNoHandle = 0xFFFFu;

struct NiQuatTransform {
    Vector3 translation;
    Quaternion rotation;
    float scale = 1.0f;

    static NiQuatTransform read(ByteReader& r) {
        NiQuatTransform t;
        t.translation.x = r.f32(); t.translation.y = r.f32(); t.translation.z = r.f32();
        t.rotation.w = r.f32(); t.rotation.x = r.f32(); t.rotation.y = r.f32(); t.rotation.z = r.f32();
        t.scale = r.f32();
        return t;
    }
    void write(ByteWriter& w) const {
        w.f32(translation.x); w.f32(translation.y); w.f32(translation.z);
        w.f32(rotation.w); w.f32(rotation.x); w.f32(rotation.y); w.f32(rotation.z);
        w.f32(scale);
    }
    static constexpr size_t kSize = 32;
};

enum class KeyType : uint32_t {
    LINEAR_KEY = 1,
    QUADRATIC_KEY = 2,
    TBC_KEY = 3,
    XYZ_ROTATION_KEY = 4,
    CONST_KEY = 5,
};

template <typename T>
struct Key {
    float time = 0;
    T value{};
    // forward/backward tangents (QUADRATIC_KEY) or TBC (TBC_KEY) are parsed
    // but discarded: this compressor refits a fresh spline through key
    // *values* only, which is an intentional, documented approximation.
};

template <typename T>
struct KeyGroup {
    KeyType interpolation = KeyType::LINEAR_KEY;
    std::vector<Key<T>> keys;
    bool present() const { return !keys.empty(); }
};

inline Vector3 readVector3(ByteReader& r) {
    Vector3 v; v.x = r.f32(); v.y = r.f32(); v.z = r.f32(); return v;
}
inline Quaternion readQuaternion(ByteReader& r) {
    Quaternion q; q.w = r.f32(); q.x = r.f32(); q.y = r.f32(); q.z = r.f32(); return q;
}

// Validates a KeyType read from the file and the key count that goes with it,
// BEFORE any container is sized. Out-of-range types would otherwise be treated
// as "no tangents" and only fail later on the block-size check; a bogus count
// would make resize(n) commit gigabytes before the per-field bounds checks
// fire. Shared by the dedupe reader so the rule is written once.
// valueBytes is the serialized size of the key value (4 float, 12 Vector3,
// 16 quaternion); QUADRATIC adds tangents (2 values, or 2 floats when the
// value is a float -- two tangents of the value type either way), TBC adds
// three floats. Quaternion keys (quadTangents=false) have no QUADRATIC
// tangents in the file format. Valid files never trip it.
inline KeyType checkKeys(ByteReader& r, uint32_t n, uint32_t rawType, size_t valueBytes, bool quadTangents = true) {
    if (rawType < 1 || rawType > 5)
        throw FormatError("Invalid KeyType " + std::to_string(rawType) + " in key data");
    KeyType t = static_cast<KeyType>(rawType);
    size_t perKey = 4 + valueBytes;
    if (t == KeyType::QUADRATIC_KEY && quadTangents) perKey += 2 * valueBytes;
    else if (t == KeyType::TBC_KEY) perKey += 12;
    r.count(n, perKey);
    return t;
}

inline KeyGroup<float> readFloatKeyGroup(ByteReader& r) {
    KeyGroup<float> g;
    uint32_t n = r.u32();
    if (n == 0) return g;
    g.interpolation = checkKeys(r, n, r.u32(), 4);
    g.keys.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        Key<float> k;
        k.time = r.f32();
        k.value = r.f32();
        if (g.interpolation == KeyType::QUADRATIC_KEY) { r.f32(); r.f32(); } // fwd/back, discarded
        else if (g.interpolation == KeyType::TBC_KEY) { r.f32(); r.f32(); r.f32(); }
        g.keys[i] = k;
    }
    return g;
}

inline KeyGroup<Vector3> readVector3KeyGroup(ByteReader& r) {
    KeyGroup<Vector3> g;
    uint32_t n = r.u32();
    if (n == 0) return g;
    g.interpolation = checkKeys(r, n, r.u32(), 12);
    g.keys.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        Key<Vector3> k;
        k.time = r.f32();
        k.value = readVector3(r);
        if (g.interpolation == KeyType::QUADRATIC_KEY) { readVector3(r); readVector3(r); }
        else if (g.interpolation == KeyType::TBC_KEY) { r.f32(); r.f32(); r.f32(); }
        g.keys[i] = k;
    }
    return g;
}

// NiTransformData / (legacy name NiKeyframeData). Rotation is either a
// QuatKey array, or (KeyType==XYZ_ROTATION_KEY) three independent float
// KeyGroups for X/Y/Z euler angles.
struct TransformData {
    uint32_t numRotationKeys = 0;
    KeyType rotationType = KeyType::LINEAR_KEY;
    std::vector<Key<Quaternion>> quatKeys;           // used if rotationType != XYZ
    KeyGroup<float> xyzRotations[3];                 // used if rotationType == XYZ
    KeyGroup<Vector3> translations;
    KeyGroup<float> scales;

    bool hasRotation() const { return numRotationKeys != 0; }
    bool isXYZRotation() const { return hasRotation() && rotationType == KeyType::XYZ_ROTATION_KEY; }

    static TransformData read(ByteReader& r, uint32_t blockSizeForValidation) {
        size_t startPos = r.pos;
        TransformData d;
        d.numRotationKeys = r.u32();
        if (d.numRotationKeys != 0) {
            uint32_t rawType = r.u32();
            if (rawType != static_cast<uint32_t>(KeyType::XYZ_ROTATION_KEY))
                d.rotationType = checkKeys(r, d.numRotationKeys, rawType, 16, false);
            else d.rotationType = KeyType::XYZ_ROTATION_KEY;
            if (d.rotationType != KeyType::XYZ_ROTATION_KEY) {
                d.quatKeys.resize(d.numRotationKeys);
                for (uint32_t i = 0; i < d.numRotationKeys; ++i) {
                    Key<Quaternion> k;
                    k.time = r.f32();
                    k.value = readQuaternion(r);
                    // QuatKey has no tangents in any interpolation mode
                    // except TBC, which stores a trailing TBC triple.
                    if (d.rotationType == KeyType::TBC_KEY) { r.f32(); r.f32(); r.f32(); }
                    d.quatKeys[i] = k;
                }
            } else {
                for (int axis = 0; axis < 3; ++axis) d.xyzRotations[axis] = readFloatKeyGroup(r);
            }
        }
        d.translations = readVector3KeyGroup(r);
        d.scales = readFloatKeyGroup(r);

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

    // Serializes an *empty* TransformData (all key counts zero) -- used to
    // shrink the original data block to near-nothing once its keys have
    // been baked into a new B-spline channel, without touching the block's
    // index (so nothing else in the file needs renumbering).
    static std::vector<uint8_t> writeEmpty() {
        ByteWriter w;
        w.u32(0); // Num Rotation Keys
        w.u32(0); // Translations.Num Keys
        w.u32(0); // Scales.Num Keys
        return w.buf;
    }
};

struct TransformInterpolator {
    NiQuatTransform transform;
    int32_t dataRef = -1;

    static TransformInterpolator read(ByteReader& r, uint32_t blockSizeForValidation) {
        size_t startPos = r.pos;
        TransformInterpolator ti;
        ti.transform = NiQuatTransform::read(r);
        ti.dataRef = r.i32();
        size_t consumed = r.pos - startPos;
        if (consumed != blockSizeForValidation) {
            throw FormatError(
                "NiTransformInterpolator: parsed " + std::to_string(consumed) +
                " bytes but block size says " + std::to_string(blockSizeForValidation) +
                " -- unsupported file variant, aborting.");
        }
        return ti;
    }
};

struct ControlledBlock {
    int32_t interpolatorRef = -1;
    int32_t controllerRef = -1;
    uint8_t priority = 0;
    int32_t nodeNameIdx = -1, propertyTypeIdx = -1, controllerTypeIdx = -1;
    int32_t controllerIdIdx = -1, interpolatorIdIdx = -1;

    static constexpr size_t kSize = 29; // 4+4+1+4*5

    static ControlledBlock read(ByteReader& r) {
        ControlledBlock cb;
        cb.interpolatorRef = r.i32();
        cb.controllerRef = r.i32();
        cb.priority = r.u8();
        cb.nodeNameIdx = r.i32();
        cb.propertyTypeIdx = r.i32();
        cb.controllerTypeIdx = r.i32();
        cb.controllerIdIdx = r.i32();
        cb.interpolatorIdIdx = r.i32();
        return cb;
    }
    void write(ByteWriter& w) const {
        w.i32(interpolatorRef);
        w.i32(controllerRef);
        w.u8(priority);
        w.i32(nodeNameIdx);
        w.i32(propertyTypeIdx);
        w.i32(controllerTypeIdx);
        w.i32(controllerIdIdx);
        w.i32(interpolatorIdIdx);
    }
};

// Only the parts of NiControllerSequence we need to locate & preserve.
struct ControllerSequence {
    int32_t nameIdx = -1;
    uint32_t arrayGrowBy = 1;
    std::vector<ControlledBlock> blocks;
    float weight = 1.0f;
    int32_t textKeysRef = -1;
    uint32_t cycleType = 0;
    float frequency = 1.0f;
    float startTime = 0, stopTime = 0;
    int32_t managerRef = -1;
    int32_t accumRootNameIdx = -1;
    // BSVersion-dependent tail (see nif.xml vercond on Anim Notes).
    enum class AnimNoteMode { None, Single, Array } animNoteMode = AnimNoteMode::None;
    int32_t animNotesRef = -1;
    std::vector<int32_t> animNoteArrayRefs;

    static ControllerSequence read(ByteReader& r, uint32_t bsVersion, uint32_t blockSizeForValidation) {
        size_t startPos = r.pos;
        ControllerSequence cs;
        cs.nameIdx = r.i32();
        uint32_t numControlled = r.u32();
        cs.arrayGrowBy = r.u32();
        r.count(numControlled, ControlledBlock::kSize);
        cs.blocks.resize(numControlled);
        for (uint32_t i = 0; i < numControlled; ++i) cs.blocks[i] = ControlledBlock::read(r);

        cs.weight = r.f32();
        cs.textKeysRef = r.i32();
        cs.cycleType = r.u32();
        cs.frequency = r.f32();
        cs.startTime = r.f32();
        cs.stopTime = r.f32();
        cs.managerRef = r.i32();
        cs.accumRootNameIdx = r.i32();

        if (bsVersion >= 24 && bsVersion <= 28) {
            cs.animNoteMode = AnimNoteMode::Single;
            cs.animNotesRef = r.i32();
        } else if (bsVersion > 28) {
            cs.animNoteMode = AnimNoteMode::Array;
            uint16_t n = r.u16();
            r.count(n, 4);
            cs.animNoteArrayRefs.resize(n);
            for (uint16_t i = 0; i < n; ++i) cs.animNoteArrayRefs[i] = r.i32();
        }

        size_t consumed = r.pos - startPos;
        if (consumed != blockSizeForValidation) {
            throw FormatError(
                "NiControllerSequence: parsed " + std::to_string(consumed) +
                " bytes but block size says " + std::to_string(blockSizeForValidation) +
                " (BSVersion=" + std::to_string(bsVersion) + ") -- unsupported file "
                "variant, aborting rather than risk corruption.");
        }
        return cs;
    }

    void write(ByteWriter& w) const {
        w.i32(nameIdx);
        w.u32(static_cast<uint32_t>(blocks.size()));
        w.u32(arrayGrowBy);
        for (auto& cb : blocks) cb.write(w);
        w.f32(weight);
        w.i32(textKeysRef);
        w.u32(cycleType);
        w.f32(frequency);
        w.f32(startTime);
        w.f32(stopTime);
        w.i32(managerRef);
        w.i32(accumRootNameIdx);
        if (animNoteMode == AnimNoteMode::Single) {
            w.i32(animNotesRef);
        } else if (animNoteMode == AnimNoteMode::Array) {
            w.u16(static_cast<uint16_t>(animNoteArrayRefs.size()));
            for (auto r_ : animNoteArrayRefs) w.i32(r_);
        }
    }
};

// ---- New block types this tool writes (never needs to *read* them, since
// input files that already use them don't need converting) ----

struct BSplineData {
    std::vector<float> floatControlPoints;
    std::vector<int16_t> compactControlPoints;

    std::vector<uint8_t> serialize() const {
        ByteWriter w;
        w.u32(static_cast<uint32_t>(floatControlPoints.size()));
        for (float v : floatControlPoints) w.f32(v);
        w.u32(static_cast<uint32_t>(compactControlPoints.size()));
        for (int16_t v : compactControlPoints) w.i16(v);
        return w.buf;
    }
};

struct BSplineBasisData {
    uint32_t numControlPoints = 0;
    std::vector<uint8_t> serialize() const {
        ByteWriter w;
        w.u32(numControlPoints);
        return w.buf;
    }
};

// Body only (Start/Stop time + Spline/Basis refs + Transform + handles,
// optionally + compact offset/range floats).
struct BSplineTransformInterpolatorOut {
    float startTime = 0, stopTime = 1;
    int32_t splineDataRef = -1, basisDataRef = -1;
    NiQuatTransform transform;
    uint32_t translationHandle = kNoHandle, rotationHandle = kNoHandle, scaleHandle = kNoHandle;
    bool compact = false;
    float translationOffset = 0, translationHalfRange = 0;
    float rotationOffset = 0, rotationHalfRange = 0;
    float scaleOffset = 0, scaleHalfRange = 0;

    std::vector<uint8_t> serialize() const {
        ByteWriter w;
        w.f32(startTime);
        w.f32(stopTime);
        w.i32(splineDataRef);
        w.i32(basisDataRef);
        transform.write(w);
        w.u32(translationHandle);
        w.u32(rotationHandle);
        w.u32(scaleHandle);
        if (compact) {
            w.f32(translationOffset); w.f32(translationHalfRange);
            w.f32(rotationOffset); w.f32(rotationHalfRange);
            w.f32(scaleOffset); w.f32(scaleHalfRange);
        }
        return w.buf;
    }
};

// ---------------- Header ----------------

struct BSStreamHeader {
    uint32_t bsVersion = 0;
    std::string author, processOrExportScript1, exportScript2, maxFilepath;
};

struct Header {
    std::string headerString;
    uint32_t version = 0;
    uint8_t endianType = 1;
    uint32_t userVersion = 0;
    uint32_t numBlocks = 0;
    bool hasBSHeader = false;
    BSStreamHeader bsHeader;
    uint16_t numBlockTypes = 0;
    std::vector<std::string> blockTypes;
    std::vector<uint16_t> blockTypeIndex; // per-block
    std::vector<uint32_t> blockSizes;     // per-block
    uint32_t numStrings = 0;
    uint32_t maxStringLength = 0;
    std::vector<std::string> strings;
    uint32_t numGroups = 0;
    std::vector<uint32_t> groups;

    int findOrAddBlockType(const std::string& name) {
        for (size_t i = 0; i < blockTypes.size(); ++i)
            if (blockTypes[i] == name) return static_cast<int>(i);
        blockTypes.push_back(name);
        numBlockTypes = static_cast<uint16_t>(blockTypes.size());
        return static_cast<int>(blockTypes.size() - 1);
    }
};

inline bool isBethesdaStreamHeaderPresent(uint32_t version, uint32_t userVersion) {
    // #BSSTREAMHEADER# condexpr from nif.xml, resolved: for our fixed
    // version 20.2.0.7 the first alternative "(VER == 20.2.0.7)" is always
    // true, so the header is present whenever userVersion (#USER#) >= 3.
    (void)version;
    return userVersion >= 3;
}

inline Header readHeader(ByteReader& r) {
    Header h;
    h.headerString = r.lineString();
    h.version = r.u32();
    h.endianType = r.u8();
    h.userVersion = r.u32();
    h.numBlocks = r.u32();

    if (h.version != kSupportedVersion) {
        // Format as real hex: std::to_string would print the decimal value
        // after the "0x" prefix, which reads like a bogus hex number.
        char hex[9];
        std::snprintf(hex, sizeof(hex), "%08X", h.version);
        throw FormatError(
            std::string("Unsupported NIF version 0x") + hex +
            " -- this tool only supports version 20.2.0.7 (0x14020007), i.e. "
            "Fallout 3 / Fallout New Vegas / late-patch Oblivion .kf files.");
    }

    // Every field after this is parsed little-endian; a big-endian file
    // (endianType 0) would be misread rather than rejected.
    if (h.endianType != 1)
        throw FormatError("Unsupported endianType " + std::to_string(h.endianType) +
                                 " -- only little-endian NIF files are supported.");

    h.hasBSHeader = isBethesdaStreamHeaderPresent(h.version, h.userVersion);
    if (h.hasBSHeader) {
        h.bsHeader.bsVersion = r.u32();
        h.bsHeader.author = r.exportString();
        // writeHeader cannot reproduce the "Unknown Int" that bsVersion > 130
        // files carry (it writes 0), so reject rather than silently drop it.
        // Never hit by Fallout 3/NV (BS version 34).
        if (h.bsHeader.bsVersion > 130)
            throw FormatError("Unsupported BS version " +
                                     std::to_string(h.bsHeader.bsVersion));
        h.bsHeader.processOrExportScript1 = r.exportString();
        h.bsHeader.exportScript2 = r.exportString();
        if (h.bsHeader.bsVersion >= 103) h.bsHeader.maxFilepath = r.exportString();
    } else {
        throw FormatError(
            "This NIF has no BSStreamHeader (not a Bethesda export) -- this "
            "tool targets Bethesda Fallout 3/NV .kf files specifically.");
    }

    h.numBlockTypes = r.u16();
    // Each count below is checked against the remaining bytes before the
    // container is sized (see ByteReader::count).
    r.count(h.numBlockTypes, 4);
    h.blockTypes.resize(h.numBlockTypes);
    for (uint16_t i = 0; i < h.numBlockTypes; ++i) h.blockTypes[i] = r.sizedString();

    r.count(h.numBlocks, 2);
    h.blockTypeIndex.resize(h.numBlocks);
    for (uint32_t i = 0; i < h.numBlocks; ++i) h.blockTypeIndex[i] = r.u16();

    // Block Size array requires version >= 20.2.0.5; guaranteed for us.
    r.count(h.numBlocks, 4);
    h.blockSizes.resize(h.numBlocks);
    for (uint32_t i = 0; i < h.numBlocks; ++i) h.blockSizes[i] = r.u32();

    h.numStrings = r.u32();
    h.maxStringLength = r.u32();
    r.count(h.numStrings, 4);
    h.strings.resize(h.numStrings);
    for (uint32_t i = 0; i < h.numStrings; ++i) h.strings[i] = r.sizedString();

    h.numGroups = r.u32();
    r.count(h.numGroups, 4);
    h.groups.resize(h.numGroups);
    for (uint32_t i = 0; i < h.numGroups; ++i) h.groups[i] = r.u32();

    return h;
}

inline void writeHeader(ByteWriter& w, const Header& h) {
    w.lineString(h.headerString);
    w.u32(h.version);
    w.u8(h.endianType);
    w.u32(h.userVersion);
    w.u32(h.numBlocks);

    w.u32(h.bsHeader.bsVersion);
    w.exportString(h.bsHeader.author);
    if (h.bsHeader.bsVersion > 130) w.u32(0);
    else w.exportString(h.bsHeader.processOrExportScript1);
    w.exportString(h.bsHeader.exportScript2);
    if (h.bsHeader.bsVersion >= 103) w.exportString(h.bsHeader.maxFilepath);

    w.u16(static_cast<uint16_t>(h.blockTypes.size()));
    for (auto& s : h.blockTypes) w.sizedString(s);
    for (auto v : h.blockTypeIndex) w.u16(v);
    for (auto v : h.blockSizes) w.u32(v);

    w.u32(h.numStrings);
    w.u32(h.maxStringLength);
    for (auto& s : h.strings) w.sizedString(s);

    w.u32(h.numGroups);
    for (auto v : h.groups) w.u32(v);
}

} // namespace nif

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
// byteio.hpp - minimal little-endian binary reader/writer over a byte buffer.
// Assumes a little-endian host (true for essentially all real-world targets:
// x86/x86-64/ARM in their default mode). NIF files are always little-endian.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Thrown for every malformed / unsupported-file condition (truncated data,
// bad counts, wrong version, size mismatches). Callers can tell "this is not
// a file we can handle" apart from genuine internal failures (bad_alloc, bugs)
// by catching this type.
struct FormatError : std::runtime_error { using std::runtime_error::runtime_error; };

struct ByteReader {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;

    ByteReader(const uint8_t* d, size_t s) : data(d), size(s) {}
    explicit ByteReader(const std::vector<uint8_t>& v) : data(v.data()), size(v.size()) {}

    void need(size_t n) const {
        // Written as n > size - pos (pos <= size is an invariant) rather than
        // pos + n > size: the sum can wrap on a 32-bit size_t and let a huge
        // n through.
        if (n > size - pos)
            throw FormatError("ByteReader: unexpected end of data (need " +
                                      std::to_string(n) + " bytes at offset " +
                                      std::to_string(pos) + ", have " +
                                      std::to_string(size - pos) + ")");
    }

    // Validates an element count read from the file BEFORE any container is
    // sized from it. A corrupt or hostile file can claim 4 billion elements;
    // resize(n) would commit tens of GB (and for 28-byte keys, much more)
    // long before the per-field bounds checks ever fire. Every serialized
    // element occupies at least minElemBytes in the stream, so if the
    // remaining bytes cannot hold n of them the file is truncated or lying
    // and we reject it up front. Valid files never trip this. Does not
    // consume anything; the caller still reads the elements.
    size_t count(uint32_t n, size_t minElemBytes) const {
        size_t remaining = size - pos;
        if (minElemBytes != 0 && n > remaining / minElemBytes)
            throw FormatError("ByteReader: element count " +
                                      std::to_string(n) + " exceeds remaining data (" +
                                      std::to_string(remaining) + " bytes at offset " +
                                      std::to_string(pos) + ")");
        return n;
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable<T>::value, "T must be POD");
        need(sizeof(T));
        T v;
        std::memcpy(&v, data + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

    uint8_t  u8()  { return read<uint8_t>(); }
    uint16_t u16() { return read<uint16_t>(); }
    uint32_t u32() { return read<uint32_t>(); }
    int32_t  i32() { return read<int32_t>(); }
    int16_t  i16() { return read<int16_t>(); }
    float    f32() { return read<float>(); }

    // Length-prefixed (uint32) string, as used for NIF "SizedString".
    std::string sizedString() {
        uint32_t len = u32();
        need(len);
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }

    // Bethesda "ExportString": 1-byte length (length INCLUDES the trailing
    // null terminator per the format spec). Distinct from SizedString --
    // conflating the two misreads every byte after it in the header.
    std::string exportString() {
        uint8_t len = u8();
        need(len);
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }

    // Newline-terminated header line ("HeaderString" / "LineString").
    std::string lineString() {
        std::string s;
        while (true) {
            need(1);
            char c = static_cast<char>(data[pos++]);
            if (c == '\n') break;
            s.push_back(c);
        }
        return s;
    }

    std::vector<uint8_t> bytes(size_t n) {
        need(n);
        std::vector<uint8_t> v(data + pos, data + pos + n);
        pos += n;
        return v;
    }

    void skip(size_t n) { need(n); pos += n; }
};

struct ByteWriter {
    std::vector<uint8_t> buf;

    template <typename T>
    void write(T v) {
        static_assert(std::is_trivially_copyable<T>::value, "T must be POD");
        size_t off = buf.size();
        buf.resize(off + sizeof(T));
        std::memcpy(buf.data() + off, &v, sizeof(T));
    }

    void u8(uint8_t v)   { write<uint8_t>(v); }
    void u16(uint16_t v) { write<uint16_t>(v); }
    void u32(uint32_t v) { write<uint32_t>(v); }
    void i32(int32_t v)  { write<int32_t>(v); }
    void i16(int16_t v)  { write<int16_t>(v); }
    void f32(float v)    { write<float>(v); }

    void sizedString(const std::string& s) {
        u32(static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    // Matches ByteReader::exportString: 1-byte length (includes the null
    // terminator that's already baked into the stored string bytes).
    void exportString(const std::string& s) {
        u8(static_cast<uint8_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void lineString(const std::string& s) {
        buf.insert(buf.end(), s.begin(), s.end());
        buf.push_back('\n');
    }

    void raw(const std::vector<uint8_t>& v) {
        buf.insert(buf.end(), v.begin(), v.end());
    }
    void raw(const uint8_t* p, size_t n) {
        buf.insert(buf.end(), p, p + n);
    }

    size_t size() const { return buf.size(); }
};

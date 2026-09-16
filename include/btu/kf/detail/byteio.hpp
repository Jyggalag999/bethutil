// byteio.hpp - minimal little-endian binary reader/writer over a byte buffer.
// Assumes a little-endian host (true for essentially all real-world targets:
// x86/x86-64/ARM in their default mode). NIF files are always little-endian.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct ByteReader {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;

    ByteReader(const uint8_t* d, size_t s) : data(d), size(s) {}
    explicit ByteReader(const std::vector<uint8_t>& v) : data(v.data()), size(v.size()) {}

    void need(size_t n) const {
        if (pos + n > size)
            throw std::runtime_error("ByteReader: unexpected end of data (need " +
                                      std::to_string(n) + " bytes at offset " +
                                      std::to_string(pos) + ", have " +
                                      std::to_string(size - pos) + ")");
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

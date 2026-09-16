/* Copyright (C) 2026 G'k
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
#pragma once

#include <btu/common/error.hpp>

namespace btu::kf {
using common::Error;

enum class KfErr : std::uint8_t
{
    Success = 0,
    Unknown = 1,
    UnsupportedFormat, // wrong NIF version, missing BSStreamHeader, or a block layout this
                       // tool's parser doesn't recognize - see convertFile()'s own messages
                       // (logged by the caller) for the specific reason.
    ConversionFailed,  // any other convertFile() failure (malformed/truncated block table,
                       // an internal fit-count invariant violation, etc.)
    TempFileIo,        // couldn't write the input span or read the output back from the
                       // temp file convertFile() itself still reads/writes by path.
};
} // namespace btu::kf

template<>
struct std::is_error_code_enum<btu::kf::KfErr> : true_type
{
}; // namespace std

namespace btu::kf {
struct KfErrCategory final : std::error_category
{
    [[nodiscard]] auto name() const noexcept -> const char * override { return "btu::kf error"; }
    [[nodiscard]] auto message(int ev) const -> std::string override
    {
        switch (static_cast<KfErr>(ev))
        {
            case KfErr::Success: return "no error";
            case KfErr::Unknown: return "default error";
            case KfErr::UnsupportedFormat: return "unsupported NIF/KF format";
            case KfErr::ConversionFailed: return "kf compression failed";
            case KfErr::TempFileIo: return "temp file read/write failed";
        }
        return "(unrecognized error)";
    };
};

inline const KfErrCategory k_kf_err_category{};
inline auto make_error_code(KfErr e) -> std::error_code
{
    return {static_cast<int>(e), k_kf_err_category};
}

enum class FailureSource : std::uint8_t
{
    BadUserInput = 1,
    SystemError  = 2,
};
} // namespace btu::kf

template<>
struct std::is_error_condition_enum<btu::kf::FailureSource> : true_type
{
}; // namespace std

namespace btu::kf {
class FailureSourceCategory final : public std::error_category
{
public:
    [[nodiscard]] auto name() const noexcept -> const char * override { return "btu::kf failure-source"; }
    [[nodiscard]] auto message(int ev) const -> std::string override
    {
        switch (static_cast<FailureSource>(ev))
        {
            case FailureSource::BadUserInput: return "invalid user request";
            case FailureSource::SystemError: return "internal error";
            default: return "(unrecognized condition)";
        }
    }
    [[nodiscard]] auto equivalent(const std::error_code &ec, int cond) const noexcept -> bool override
    {
        switch (static_cast<FailureSource>(cond))
        {
            case FailureSource::SystemError: return ec == KfErr::Unknown;
            case FailureSource::BadUserInput: return ec == KfErr::UnsupportedFormat;
            default: return false;
        }
    }
};

inline const FailureSourceCategory k_failure_source_category{};
inline auto make_error_condition(FailureSource e) -> std::error_condition
{
    return {static_cast<int>(e), k_failure_source_category};
}
} // namespace btu::kf

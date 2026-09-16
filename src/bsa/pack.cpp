/* Copyright (C) 2021 Edgar B
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#include "btu/bsa/pack.hpp"

#include "btu/bsa/archive.hpp"
#include "btu/bsa/settings.hpp"

#include <btu/common/algorithms.hpp>
#include <btu/common/functional.hpp>
#include <btu/common/threading.hpp>
#include <flux.hpp>

#include <functional>

namespace btu::bsa {
auto get_allow_file_pred(const PackSettings &sets) -> AllowFilePred
{
    // this part is required, these files would break the archive if packed
    auto is_legal_path = [](const Path &root_dir, const fs::directory_entry &fileinfo) -> bool {
        const bool is_regular = is_regular_file(fileinfo);

        //Removing files at the root directory, those cannot be packed
        const bool is_at_root = equivalent(fileinfo.path().parent_path(), root_dir);

        return is_regular && !is_at_root;
    };

    // now we add the user predicate
    return [legal = BTU_MOV(is_legal_path),
            sets  = BTU_MOV(sets)](const Path &root_dir, const fs::directory_entry &fileinfo) -> bool {
        const bool user_allowed = !sets.allow_file_pred || (*sets.allow_file_pred)(root_dir, fileinfo);

        return legal(root_dir, fileinfo) && user_allowed;
    };
}

struct PackGroup
{
    std::vector<Path> standard;
    std::vector<Path> texture;
    std::vector<Path> mesh;
    std::vector<Path> sound;
};

/// \brief Decide which physical archive a file belongs in, reusing the tes4_archive_type
/// classification that already exists per-extension in Settings (standard_files / texture_files /
/// incompressible_files). Only splits into a dedicated archive if the corresponding
/// has_*_version flag is enabled; otherwise everything not routed to Textures falls back to Standard.
[[nodiscard]] auto classify_archive_bucket(const Path &file_path,
                                           const Settings &sets,
                                           FileTypes filetype) noexcept -> ArchiveType
{
    // Textures keep using the existing FileTypes-based split (unchanged behavior).
    if (sets.has_texture_version && filetype == FileTypes::Texture)
        return ArchiveType::Textures;

    const auto tes4_type = get_tes4_archive_type(file_path, sets);
    if (!tes4_type)
        return ArchiveType::Standard;

    if (sets.has_mesh_version && *tes4_type == TES4ArchiveType::meshes)
        return ArchiveType::Meshes;

    if (sets.has_sound_version
        && (*tes4_type == TES4ArchiveType::sounds || *tes4_type == TES4ArchiveType::voices))
        return ArchiveType::Sounds;

    return ArchiveType::Standard;
}

/// \brief List all files in the directory which can be packed, sorted by size (largest first)
[[nodiscard]] auto list_packable_files(const Path &dir,
                                       const Settings &sets,
                                       const AllowFilePred &allow_path_pred) noexcept -> PackGroup
{
    constexpr std::array allowed_types = {FileTypes::Standard, FileTypes::Texture, FileTypes::Incompressible};

    auto packable_files = flux::from_range(fs::recursive_directory_iterator(dir))
                              .filter([&](const auto &p) {
                                  // filter out empty files
                                  return p.is_regular_file() && p.file_size() > 0 && allow_path_pred(dir, p)
                                         && common::contains(allowed_types, get_filetype(p, dir, sets));
                              })
                              .map([](const auto &p) { return p.path(); })
                              .to<std::vector>();

    // sort by size, largest first
    std::ranges::sort(packable_files, [](const auto &lhs, const auto &rhs) {
        return fs::file_size(lhs) > fs::file_size(rhs);
    });

    // distribute into buckets, preserving the largest-first order within each one
    PackGroup groups;
    for (auto &file : packable_files)
    {
        const auto filetype = get_filetype(file, dir, sets);
        switch (classify_archive_bucket(file, sets, filetype))
        {
            case ArchiveType::Textures: groups.texture.push_back(std::move(file)); break;
            case ArchiveType::Meshes: groups.mesh.push_back(std::move(file)); break;
            case ArchiveType::Sounds: groups.sound.push_back(std::move(file)); break;
            case ArchiveType::Standard: groups.standard.push_back(std::move(file)); break;
        }
    }

    return groups;
}

[[nodiscard]] auto prepare_file(const Path &file_path,
                                const PackSettings &sets,
                                const ArchiveType type) noexcept -> std::optional<File>
{
    auto file = File{sets.game_settings.version, type, get_tes4_archive_type(file_path, sets.game_settings)};
    const bool read_success = file.read(file_path);
    if (!read_success)
        return std::nullopt;

    const bool dx = (file.version() == ArchiveVersion::fo4 || file.version() == ArchiveVersion::starfield)
                    && type == ArchiveType::Textures;

    const bool compressible = get_filetype(file_path, sets.input_dir, sets.game_settings)
                              != FileTypes::Incompressible;

    if ((sets.compress == Compression::Yes && compressible) || dx) // dx is always compressed
    {
        const bool compress_success = file.compress();
        if (!compress_success && dx) // we only care about failure if it's a texture archive
            return std::nullopt;
    }
    return file;
}

[[nodiscard]] auto file_fits(const Archive &arch, const File &file, const Settings &sets) noexcept -> bool
{ return arch.file_size() + file.size().value_or(0) <= sets.max_size; }

[[nodiscard]] auto do_pack(std::vector<Path> file_paths,
                           const PackSettings settings,
                           const ArchiveType type) noexcept -> flux::generator<Archive &&>
{
    // NOTE: gcc appears to have issues with structured bindings in coroutines, as
    // using it here produces a "may be used uninitialized" warning
    auto producer = common::make_producer_mt<std::optional<Archive::value_type>>(
        std::move(file_paths), [&](const Path &absolute_path) -> std::optional<Archive::value_type> {
            return prepare_file(absolute_path, settings, type)
                .transform([&](File &&file) -> Archive::value_type {
                    return {relative(absolute_path, settings.input_dir).string(), std::move(file)};
                });
        });

    [[maybe_unused]] auto thread = BTU_MOV(producer.first);
    auto receiver                = BTU_MOV(producer.second);

    auto make_arch = [&settings, type] { return Archive{settings.game_settings.version, type}; };
    auto arch      = make_arch();

    for (auto &&maybe_prepared : receiver)
    {
        if (!maybe_prepared.has_value())
            continue; // just ignore this file. TODO: maybe warn?

        auto prepared             = BTU_MOV(maybe_prepared).value();
        const auto &relative_path = prepared.first;
        auto &file                = prepared.second;

        if (file_fits(arch, file, settings.game_settings))
        {
            const bool success = arch.emplace(BTU_MOV(relative_path), BTU_MOV(file));
            assert(success && "file type in bsa mismatch, this should not happen");
            continue;
        }

        // if we are here, the file does not fit into the archive,
        // so we yield the current archive and start a new one
        co_yield std::exchange(arch, make_arch());

        const bool success = arch.emplace(BTU_MOV(relative_path), BTU_MOV(file));
        assert(success && "file type in bsa mismatch, this should not happen");

        // is it even possible that the file is bigger than the max size?
        assert(arch.file_size() <= settings.game_settings.max_size);
    }

    // return the last archive
    if (!arch.empty())
        co_yield BTU_MOV(arch);
}

auto pack(const PackSettings settings) noexcept -> flux::generator<Archive &&>
{
    auto groups = list_packable_files(settings.input_dir,
                                      settings.game_settings,
                                      get_allow_file_pred(settings));

    if (!groups.standard.empty())
    {
        FLUX_FOR(auto &&a, do_pack(BTU_MOV(groups.standard), settings, ArchiveType::Standard))
        { co_yield BTU_MOV(a); }
    }

    if (!groups.texture.empty())
    {
        FLUX_FOR(auto &&a, do_pack(BTU_MOV(groups.texture), settings, ArchiveType::Textures))
        { co_yield BTU_MOV(a); }
    }

    if (!groups.mesh.empty())
    {
        FLUX_FOR(auto &&a, do_pack(BTU_MOV(groups.mesh), settings, ArchiveType::Meshes))
        { co_yield BTU_MOV(a); }
    }

    if (!groups.sound.empty())
    {
        FLUX_FOR(auto &&a, do_pack(BTU_MOV(groups.sound), settings, ArchiveType::Sounds))
        { co_yield BTU_MOV(a); }
    }
}

} // namespace btu::bsa

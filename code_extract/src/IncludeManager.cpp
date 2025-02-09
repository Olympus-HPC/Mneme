#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utils/IncludeManager.h"

IncludeManager::IncludeManager(
    std::string const &directory,
    std::unordered_set<std::string> const &includePaths) {
  for (auto incPath : includePaths) {
    if (incPath.find(directory) != std::string::npos)
      include_paths.push_back(incPath);
  }

  processDirectory(directory);

  resolve();
}

void IncludeManager::getIncludesFromFile(FileID id, bool isHeader) {
  auto path = id_to_path[id];
  std::ifstream file(path);

  std::string line;
  std::regex include_regex(R"(#include\s+["<](.*)[">])");

  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_match(line, match, include_regex)) {
      auto incFile = match[1].str();
      FileID includeID;
      if (isHeader) {
        includeID = addFile(incFile, true);
        header_includes[id].insert(includeID);
      } else {
        includeID = addFile(incFile, true, path.substr(0, path.rfind('/')));
        source_includes[id].insert(includeID);
      }

      if (!id_to_isExternal[includeID])
        getIncludesFromFile(includeID, true);
    }
  }
}

void IncludeManager::resolve(FileID file,
                             std::unordered_set<FileID> &includes) {
  if (id_to_resolve[file])
    return;

  auto set = includes;
  includes.clear();
  for (auto id : set) {
    if (id_to_isExternal[id])
      includes.insert(id);
    else {
      resolve(id, header_includes[id]);
      includes.insert(header_includes[id].begin(), header_includes[id].end());
    }
  }

  id_to_resolve[file] = true;
}

void IncludeManager::resolve() {
  for (auto &[id, includes] : source_includes)
    resolve(id, includes);
}

void IncludeManager::processDirectory(std::string const &directory) {
  for (auto const &entry :
       std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      auto const &file = entry.path();
      auto ext = file.extension().string();
      if (fileExt.find(ext) != fileExt.end()) {
        auto id = addFile(file);
        getIncludesFromFile(id, false);
      }
    }
  }
}

IncludeManager::FileID IncludeManager::addFile(std::string const &file,
                                               bool isHeader,
                                               std::string const &baseSrc) {
  auto it = path_to_id.find(file);
  if (it == path_to_id.end()) {
    path_to_id[file] = id_to_path.size();
    id_to_path.push_back(file);

    auto id = id_to_path.size() - 1;
    if (isHeader) {
      header_includes[id] = {};
      id_to_isExternal.push_back(isIncludeExternal(file, baseSrc));
      id_to_resolve.push_back(id_to_isExternal.back());
    } else {
      source_includes[id] = {};
      id_to_isExternal.push_back(false);
      id_to_resolve.push_back(false);
    }

    return id;
  } else
    return it->second;
}

bool IncludeManager::isIncludeExternal(std::string const &incFile,
                                       std::string const &baseSrc) const {
  for (auto const &path : include_paths) {
    if (std::filesystem::exists(path + '/' + incFile))
      return false;
  }

  return !std::filesystem::exists(baseSrc + '/' + incFile);
}

std::string IncludeManager::getFileFromID(FileID id) const {
  if (id >= id_to_path.size())
    return "";
  else
    return id_to_path[id];
}

IncludeManager::FileID
IncludeManager::getIDFromFile(std::string const &file) const {
  auto it = path_to_id.find(file);
  if (it != path_to_id.end())
    return it->second;

  for (auto const &path : include_paths) {
    auto idx = file.find(path);
    if (idx != std::string::npos)
      return path_to_id.at(file.substr(idx + path.size() + 1));
  }

  assert(false && "All included files should already be seen!");
  return -1;
}

void IncludeManager::getIncludes(FileID id,
                                 std::unordered_set<FileID> &outSet) const {
  auto src = source_includes.find(id);
  if (src != source_includes.end()) {
    outSet.insert(src->second.begin(), src->second.end());
  } else {
    auto &header = header_includes.at(id);
    assert(id_to_resolve[id] && "All headers to visit must be resolved!");
    outSet.insert(header.begin(), header.end());
  }
}

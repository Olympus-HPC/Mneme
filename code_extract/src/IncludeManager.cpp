#include <cassert>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utils/IncludeManager.h"
#include <iostream>

IncludeManager::IncludeManager(
    std::string const &directory,
    std::unordered_set<std::string> const &includePaths) {
  for (auto incPath : includePaths) {
    if (incPath.find(directory) != std::string::npos)
      include_paths.push_back(incPath);
  }
  include_paths.push_back(directory);
  
  processDirectory(directory);

  resolve();

  // In an ideal world, we should not need this...
  resolve_transitivity();
}

void IncludeManager::getIncludesFromFile(FileID id, std::string const &fullPath,
                                         bool isHeader) {
  std::string currDir = fullPath.substr(0, fullPath.rfind('/'));
  std::ifstream file(fullPath);

  std::string line;
  std::regex include_regex(R"(#include\s+["<](.*)[">]\s*$)");

  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_match(line, match, include_regex)) {

      auto incFile = match[1].str();
      FileID includeID = addFile(incFile, true, currDir);
      
      auto [isExternal, path] = id_to_isExternal[includeID];
      if (!isExternal){
        transitive_includes[includeID] = id;
      }

      // If we have already gotten includes from this before, skip
      if (includeID >= id_to_path.size())
        continue;
          
      if (isHeader)
        header_includes[id].insert(includeID);
      else
        source_includes[id].insert(includeID);
      
      if (!isExternal)
        getIncludesFromFile(includeID, path, true);
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
    if (id_to_isExternal[id].first)
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

void IncludeManager::resolve_transitivity(FileID id, std::unordered_set<FileID>* include_to, std::vector<bool>& visited) {
  resolve_transitivity(id, visited);

  std::unordered_set<FileID>* to_include = nullptr;

  auto src = source_includes.find(id);
  if (src != source_includes.end())
    to_include = &src->second;
  else
    to_include = &header_includes.at(id);

  include_to->insert(to_include->begin(), to_include->end());
}

void IncludeManager::resolve_transitivity(FileID id, std::vector<bool>& visited) {
  if (visited[id]) return;

  auto it = transitive_includes.find(id);
  if (it == transitive_includes.end()) return;
  auto includee = it->second;

  std::unordered_set<FileID>* include_to = nullptr;
  auto src = source_includes.find(id);
  if (src != source_includes.end())
    include_to = &src->second;
  else
    include_to = &header_includes.at(id);

  resolve_transitivity(includee, include_to, visited);
  visited[id] = true;
}

void IncludeManager::resolve_transitivity() {
  std::vector<bool> visited(id_to_path.size(), false);
  for (auto &[id, includee] : transitive_includes) {
    resolve_transitivity(id, visited);
  }
}

void IncludeManager::processDirectory(std::string const &directory) {
  for (auto const &entry :
       std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      auto const &file = entry.path();
      auto ext = file.extension().string();
      if (fileExt.find(ext) != fileExt.end()) {
        auto id = addFile(file.lexically_relative(directory));
        getIncludesFromFile(id, file, false);
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
    auto [isExternal, path] = isIncludeExternal(file, baseSrc);
    if (isHeader) {
      header_includes[id] = {};
      id_to_isExternal.emplace_back(isExternal, path);
      id_to_resolve.push_back(isExternal);
    } else {
      source_includes[id] = {};
      id_to_isExternal.emplace_back(false, path);
      id_to_resolve.push_back(false);
    }

    return id;
  } else
    return it->second;
}

std::tuple<bool, std::string>
IncludeManager::isIncludeExternal(std::string const &incFile,
                                  std::string const &baseSrc) {
  // Check if it is in any of our include paths
  for (auto const &path : include_paths) {
    auto currPath = path + '/' + incFile;
    if (std::filesystem::exists(currPath))
      return {false, currPath};
  }

  // Then check if it is in the source local path, 
  // ideally this should already be covered by the previous check.
  auto localPath = baseSrc + '/' + incFile;
  if (std::filesystem::exists(localPath)) {
    include_paths.push_back(baseSrc);
    return {false, localPath};
  } else
    return {true, ""};
}

std::string IncludeManager::getFileFromID(FileID id) const {
  if (id >= id_to_path.size())
    return "";
  else
    return id_to_path[id];
}

std::tuple<int, bool>
IncludeManager::getIDFromFile(std::string const &file) const {
  auto it = path_to_id.find(file);
  if (it != path_to_id.end())
    return {it->second, true};

  // If it is in our include path, try to look it up in our header db.
  for (auto const &path : include_paths) {
    auto idx = file.find(path);
    if (idx != std::string::npos)
      return {path_to_id.at(file.substr(idx + path.size() + 1)), true};
  }

  return {-1, false};
}

void IncludeManager::getIncludes(FileID id,
                                 std::unordered_set<FileID> &outSet) const {
  auto src = source_includes.find(id);
  if (src != source_includes.end())
    outSet.insert(src->second.begin(), src->second.end());
  else {
    auto &header = header_includes.at(id);
    assert(id_to_resolve[id] && "All headers to visit must be resolved!");
    outSet.insert(header.begin(), header.end());
  }
}

void IncludeManager::getAllExternals(std::unordered_set<FileID> &outSet) const {
  for (std::size_t idx = 0; idx < id_to_isExternal.size(); idx++)
    if (id_to_isExternal[idx].first)
      outSet.insert(idx);
}
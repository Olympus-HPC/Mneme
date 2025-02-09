#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class IncludeManager {
  using FileID = std::size_t;
  std::unordered_map<FileID, std::unordered_set<FileID>> source_includes;
  std::unordered_map<FileID, std::unordered_set<FileID>> header_includes;

  std::vector<std::string> id_to_path;
  std::vector<bool> id_to_isExternal;
  std::vector<bool> id_to_resolve;
  std::unordered_map<std::string, FileID> path_to_id;

  std::vector<std::string> include_paths;
  std::unordered_set<std::string> const fileExt = {".c", ".cu", ".cpp"};

  void getIncludesFromFile(FileID id, bool isHeader);

  void resolve(FileID file, std::unordered_set<FileID> &includes);

  void resolve();

  void processDirectory(std::string const &directory);

  FileID addFile(std::string const &file, bool isHeader = false, std::string const& baseSrc = "");

public:
  IncludeManager(std::string const &directory,
                 std::unordered_set<std::string> const &includePaths);

  bool isIncludeExternal(std::string const &incFile, std::string const& baseSrc) const;

  std::string getFileFromID(FileID id) const;

  FileID getIDFromFile(std::string const &file) const;

  void getIncludes(FileID id, std::unordered_set<FileID> &outSet) const;
};
#pragma once
#include <map>
#include <string>

namespace splatx {

// Mini JSON parser for querying meta.json (numbers only, flattened).
// Nested objects are returned as flattened keys of the form "tolerances.means2d.rtol",
// and non-numeric values (strings, booleans, null, and arrays) are ignored.
// A missing file or a syntax error raises std::runtime_error.
std::map<std::string, double> json_parse_flat(const std::string& path);

} // namespace splatx

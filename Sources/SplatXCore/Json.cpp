#include "splatx/Json.h"
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace splatx {
namespace {

// Recursive-descent parser dedicated to meta.json. It collects only numeric values, under flattened
// keys ("a.b.c"); strings, booleans, nulls, and arrays are syntax-checked only and skipped.
struct Parser {
    const std::string& s;
    size_t i = 0;
    std::map<std::string, double> out;
    bool collect = true; // false inside an array — even if an element is an object, its numeric members are not collected

    [[noreturn]] void fail(const char* msg) {
        throw std::runtime_error("json_parse_flat: " + std::string(msg) +
                                 " at offset " + std::to_string(i));
    }
    void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++; }
    char peek() { ws(); if (i >= s.size()) fail("unexpected end"); return s[i]; }
    void expect(char c) { if (peek() != c) fail("unexpected character"); i++; }

    std::string string_lit() {
        expect('"');
        std::string r;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') { i++; if (i >= s.size()) fail("bad escape"); }
            r += s[i++];
        }
        if (i >= s.size()) fail("unterminated string");
        i++;
        return r;
    }
    void literal(const char* lit) {
        for (const char* p = lit; *p; p++) { if (i >= s.size() || s[i] != *p) fail("bad literal"); i++; }
    }
    void number(const std::string& key) {
        const char* start = s.c_str() + i;
        char* end = nullptr;
        double v = std::strtod(start, &end);
        if (end == start) fail("expected value");
        i += static_cast<size_t>(end - start);
        if (collect && !key.empty()) out[key] = v;
    }
    // Consumes one value. If it is a number and key is non-empty, records it into out[key].
    void value(const std::string& key) {
        char c = peek();
        if (c == '{') object(key);
        else if (c == '[') array();
        else if (c == '"') string_lit();
        else if (c == 't') literal("true");
        else if (c == 'f') literal("false");
        else if (c == 'n') literal("null");
        else number(key);
    }
    void object(const std::string& prefix) {
        expect('{');
        if (peek() == '}') { i++; return; }
        while (true) {
            ws();
            std::string k = string_lit();
            expect(':');
            value(prefix.empty() ? k : prefix + "." + k);
            char c = peek();
            i++;
            if (c == ',') continue;
            if (c == '}') return;
            fail("expected ',' or '}'");
        }
    }
    void array() {
        expect('[');
        const bool saved = collect;
        collect = false;  // array elements are syntax-checked only — prevents members of object elements from leaking (JsonTests regression)
        if (peek() == ']') { i++; collect = saved; return; }
        while (true) {
            value("");
            char c = peek();
            i++;
            if (c == ',') continue;
            if (c == ']') { collect = saved; return; }
            fail("expected ',' or ']'");
        }
    }
};

} // namespace

std::map<std::string, double> json_parse_flat(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("json_parse_flat: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    Parser p{text};
    p.object("");  // the root must be an object
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return std::move(p.out);
}

} // namespace splatx

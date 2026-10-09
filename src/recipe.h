// The photo's settings as RAWmakase saved them: read from its catalog, read-only, so
// the POC starts from the same edit and uses the profile tone curve RAWmakase renders
// with. Adobe's profile data is never stored by the POC.
#pragma once

#include <sqlite3.h>

#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace recipe {

// A minimal JSON value: enough to read a recipe.
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double number = 0;
    bool boolean = false;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json* get(const std::string& key) const {
        auto it = object.find(key);
        return kind == Object && it != object.end() ? &it->second : nullptr;
    }
    double num(const std::string& key, double fallback) const {
        const Json* v = get(key);
        return v && v->kind == Number ? v->number : fallback;
    }
};

class Parser {
  public:
    explicit Parser(const std::string& text) : s_(text) {}
    Json parse() {
        Json v = value();
        space();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

  private:
    const std::string& s_;
    size_t i_ = 0;
    [[noreturn]] void fail(const char* what) {
        throw std::runtime_error(std::string("recipe JSON: ") + what);
    }
    void space() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    bool take(char c) {
        space();
        if (i_ < s_.size() && s_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }
    std::string str() {
        if (!take('"')) fail("expected string");
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                char e = s_[++i_];
                out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                if (e == 'u') i_ += 4; // code points are not needed here
            } else {
                out += s_[i_];
            }
            ++i_;
        }
        ++i_;
        return out;
    }
    Json value() {
        space();
        if (i_ >= s_.size()) fail("unexpected end");
        Json v;
        char c = s_[i_];
        if (c == '{') {
            ++i_;
            v.kind = Json::Object;
            if (take('}')) return v;
            do {
                std::string key = str();
                if (!take(':')) fail("expected ':'");
                v.object[key] = value();
            } while (take(','));
            if (!take('}')) fail("expected '}'");
        } else if (c == '[') {
            ++i_;
            v.kind = Json::Array;
            if (take(']')) return v;
            do v.array.push_back(value());
            while (take(','));
            if (!take(']')) fail("expected ']'");
        } else if (c == '"') {
            v.kind = Json::String;
            v.string = str();
        } else if (s_.compare(i_, 4, "true") == 0) {
            v.kind = Json::Bool;
            v.boolean = true;
            i_ += 4;
        } else if (s_.compare(i_, 5, "false") == 0) {
            v.kind = Json::Bool;
            i_ += 5;
        } else if (s_.compare(i_, 4, "null") == 0) {
            i_ += 4;
        } else {
            char* end = nullptr;
            v.kind = Json::Number;
            v.number = std::strtod(s_.c_str() + i_, &end);
            if (end == s_.c_str() + i_) fail("unexpected character");
            i_ = size_t(end - s_.c_str());
        }
        return v;
    }
};

// What the POC takes from the saved recipe.
struct Settings {
    bool found = false;
    std::string source;          // catalog it came from
    double exposure = 0, contrast = 0, highlights = 0, shadows = 0, whites = 0, blacks = 0;
    double camera_exposure = 0;  // the DNG's baseline exposure, as RAWmakase applies it
    double profile_exposure = 0; // the camera profile's own exposure
    std::string profile;
    std::vector<float> tone_x, tone_y; // the profile tone curve (linear ProPhoto)
};

// Reads the catalog row whose original path is `photo`.
inline Settings load(const std::string& catalog, const std::string& photo) {
    Settings out;
    sqlite3* db = nullptr;
    std::string uri = "file:" + catalog + "?mode=ro";
    if (sqlite3_open_v2(uri.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return out;
    }
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> guard(db, sqlite3_close);
    sqlite3_stmt* st = nullptr;
    const char* sql = "SELECT recipe FROM photos WHERE original_path = ? AND recipe IS NOT NULL "
                      "ORDER BY copy_name = '' DESC LIMIT 1";
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) return out;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(st, sqlite3_finalize);
    sqlite3_bind_text(st, 1, photo.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st) != SQLITE_ROW) return out;
    std::string text(reinterpret_cast<const char*>(sqlite3_column_text(st, 0)),
                     size_t(sqlite3_column_bytes(st, 0)));
    Json r = Parser(text).parse();
    out.found = true;
    out.source = catalog;
    out.exposure = r.num("exposure", 0);
    out.contrast = r.num("contrast", 0);
    out.highlights = r.num("highlights", 0);
    out.shadows = r.num("shadows", 0);
    out.whites = r.num("whites", 0);
    out.blacks = r.num("blacks", 0);
    out.camera_exposure = r.num("camera_exposure", 0);
    if (const Json* p = r.get("profile")) {
        if (const Json* name = p->get("name")) out.profile = name->string;
        out.profile_exposure = p->num("exposure", 0);
        if (const Json* tone = p->get("tone"))
            for (const Json& point : tone->array)
                if (point.array.size() == 2) {
                    out.tone_x.push_back(float(point.array[0].number));
                    out.tone_y.push_back(float(point.array[1].number));
                }
    }
    return out;
}

} // namespace recipe

// JSON Schema -> GBNF grammar (common/gbnf.h). Adapted from llama.cpp's
// json-schema-to-grammar; see the .cpp for provenance.

#pragma once

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

namespace luce::common {

// The GBNF grammar whose `root` rule matches exactly the JSON documents the
// schema describes (with the whitespace llama.cpp allows between tokens).
// Throws std::runtime_error when the schema cannot be converted.
std::string json_schema_to_grammar(const nlohmann::ordered_json & schema);

// Compose a grammar from schemas and hand-written rules: add_schema returns
// the rule name for a schema, add_rule adds (or reuses) a named rule.
struct GrammarBuilder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const nlohmann::ordered_json &)> add_schema;
    std::function<void(nlohmann::ordered_json &)> resolve_refs;
};

struct GrammarOptions {
    bool dotall = false;
};

std::string build_grammar(const std::function<void(const GrammarBuilder &)> & cb,
                          const GrammarOptions & options = {});

}  // namespace luce::common

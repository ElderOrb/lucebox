// GBNF grammars: the parser and the pushdown matcher that constrained
// decoding runs on (output_constraint.h).
//
// Adapted from llama.cpp src/llama-grammar.{h,cpp} (release b6500),
// MIT License, Copyright (c) 2023-2024 The ggml authors. The grammar syntax
// is llama.cpp's own (grammars/README.md upstream), so a grammar written for
// llama-server works here unchanged. Left out: lazy trigger words and the
// llama_vocab coupling; tokens are matched by their decoded bytes.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace luce::common::gbnf {

enum class ElementType : uint8_t {
    END,             // end of rule definition
    ALT,             // start of an alternate definition of the rule
    RULE_REF,        // non-terminal: reference to a rule
    CHAR,            // terminal: code point
    CHAR_NOT,        // inverse char(s): [^a], [^a-b], [^abc]
    CHAR_RNG_UPPER,  // makes the preceding CHAR/CHAR_ALT an inclusive range: [a-z]
    CHAR_ALT,        // adds an alternate char to the preceding CHAR/range: [ab], [a-zA]
    CHAR_ANY,        // any character: .
};

struct Element {
    ElementType type;
    uint32_t    value;  // code point or rule id
};

struct PartialUtf8 {
    uint32_t value    = 0;  // bit value so far (unshifted)
    int      n_remain = 0;  // bytes still missing; -1 marks an invalid sequence
};

using Rule   = std::vector<Element>;
using Rules  = std::vector<Rule>;
using Stack  = std::vector<const Element *>;
using Stacks = std::vector<Stack>;

// Decode `bytes` as UTF-8, continuing a sequence an earlier piece left open.
// The code points end with a 0 terminator; the second value is the sequence
// this piece leaves open (n_remain 0 when it ends on a character boundary).
std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8(
        const std::string & bytes, PartialUtf8 start);

struct Parser {
    std::map<std::string, uint32_t> symbol_ids;
    Rules rules;
    std::string error;

    // False (with `error` set and `rules` cleared) when the text is not a
    // well-formed grammar or references an undefined rule.
    bool parse(const char * src);
    void print(FILE * file) const;

private:
    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);
    void add_rule(uint32_t rule_id, const Rule & rule);
    const char * parse_alternates(const char * src, const std::string & rule_name,
                                  uint32_t rule_id, bool is_nested);
    const char * parse_sequence(const char * src, const std::string & rule_name,
                                Rule & rule, bool is_nested);
    const char * parse_rule(const char * src);
};

// A token under test: its decoded code points (0-terminated) from the
// matcher's current partial UTF-8 state. `index` is the caller's own id.
struct Candidate {
    size_t           index;
    const uint32_t * code_points;
    PartialUtf8      partial;
};

// One position in a grammar: the rules (shared, immutable) and the pushdown
// stacks every way of reaching this position leaves. Copies share the rules,
// so a copy is a cheap snapshot of the position.
class Matcher {
public:
    // Parse `text` and start at rule `root`. Null with `error` set on a parse
    // failure, a missing root, or left recursion.
    static std::unique_ptr<Matcher> compile(const std::string & text,
                                            const std::string & root,
                                            std::string & error);

    // True when the text so far is a complete sentence of the grammar.
    bool can_end() const;
    // True when no continuation is possible (the last piece did not match).
    bool dead() const { return stacks_.empty(); }
    PartialUtf8 partial() const { return partial_; }
    const Stacks & stacks() const { return stacks_; }
    const Rules & rules() const { return *rules_; }
    // Id of a named rule, -1 when absent.
    int rule_id(const std::string & name) const;

    // Advance over `bytes`. On a mismatch the matcher becomes dead().
    void accept_bytes(const std::string & bytes);

    // Every candidate that cannot continue the text from this position.
    std::vector<Candidate> reject(const std::vector<Candidate> & candidates) const;

private:
    Matcher() = default;

    std::shared_ptr<const Rules> rules_;
    std::map<std::string, uint32_t> symbol_ids_;
    Stacks stacks_;
    PartialUtf8 partial_;
};

}  // namespace luce::common::gbnf

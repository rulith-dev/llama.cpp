#pragma once

#include "json.h"

#include <functional>
#include <memory>
#include <string>

std::string json_schema_to_grammar(const common_json & schema,
                                   bool force_gbnf = false);

class common_schema_converter;

// strixllama: a tool's parameters schema with its properties and required list spelled out at the top level, however
// the schema states them: a top-level $ref (to #/$defs or #/definitions), allOf parts (each requiring what it requires),
// oneOf / anyOf alternatives (offered together, required only where every alternative requires them) and
// {not: {required: [...]}} (those left out). A schema that already lists its properties plainly comes back unchanged.
// For the formats that build one grammar rule per parameter; what the model is shown is not affected.
common_json common_tool_parameters_flatten(const common_json & params);

// Probes a JSON schema to extract information about its structure and type constraints.
class common_schema_info {
    std::unique_ptr<common_schema_converter> impl_;

  public:
    common_schema_info();
    ~common_schema_info();

    common_schema_info(const common_schema_info &) = delete;
    common_schema_info & operator=(const common_schema_info &) = delete;
    common_schema_info(common_schema_info &&) noexcept;
    common_schema_info & operator=(common_schema_info &&) noexcept;

    void resolve_refs(common_json & schema);
    bool resolves_to_string(const common_json & schema);
};

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const common_json &)> add_schema;
    std::function<void(common_json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});

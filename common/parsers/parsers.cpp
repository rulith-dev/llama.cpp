#include "parsers.h"
#include "json-schema-to-grammar.h"

#include "log.h"

#include <map>
#include <set>
#include <string>
#include <vector>

void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

void foreach_parameter(const json & function, const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    // strixllama: the parameters as one flat list whatever form the schema takes (common_tool_parameters_flatten): a
    // top-level oneOf, allOf or $ref had no parameter here, so the grammar allowed an empty call and nothing else
    const json params = common_tool_parameters_flatten(function.at("parameters"));
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto & props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        required = params.at("required").get<std::set<std::string>>();
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}

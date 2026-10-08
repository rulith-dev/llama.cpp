#include "parsers.h"

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
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        // strixllama: parameters given only as oneOf / anyOf alternatives of objects (Rulith's OpenCase: caseType with
        // businessKey, or caseId) had no parameter at all here, so the tool-call grammar allowed an empty call and
        // nothing else - every call arrived as {}. Every alternative's properties are offered, each required only where
        // all alternatives require it; which combination is valid stays the tool's to check
        const char * key = params.contains("oneOf") ? "oneOf" : params.contains("anyOf") ? "anyOf" : nullptr;
        if (!key || !params.at(key).is_array()) {
            return;
        }
        std::vector<std::string>        order;
        std::map<std::string, json>     props;
        std::map<std::string, size_t>   n_required;
        size_t n_alt = 0;
        for (const auto & alt : params.at(key)) {
            if (!alt.is_object() || !alt.contains("properties") || !alt.at("properties").is_object()) {
                continue;
            }
            ++n_alt;
            for (const auto & [name, prop] : alt.at("properties").items()) {
                if (props.find(name) == props.end()) {
                    order.push_back(name);
                    props[name] = prop;
                }
            }
            if (alt.contains("required") && alt.at("required").is_array()) {
                for (const auto & r : alt.at("required")) {
                    if (r.is_string()) {
                        n_required[r.get<std::string>()]++;
                    }
                }
            }
        }
        for (const auto & name : order) {
            fn(name, props[name], n_alt > 0 && n_required[name] == n_alt);
        }
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

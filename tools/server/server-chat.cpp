#include "server-chat.h"
#include "server-common.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    // strixllama: the tools first, so calls in the input can use the names the model is shown (coding agents such as
    // Codex send custom and namespace tools; they were dropped and every round trip of them failed). Tools come from
    // "tools" and from additional_tools input items. Function tools pass through; a custom (freeform) tool becomes a
    // function taking one string "input"; the tools inside a namespace become functions of their own, named as they
    // are when the name is unique and NAMESPACE__NAME otherwise. Tools the server cannot run (web_search, file_search,
    // ...) are left out and listed in strix_resp_unsupported (the X-Rulith-Unsupported-Tools response header).
    // strix_resp_tools tells the response side how to turn a call back into what the client offered.
    struct resp_tool {
        std::string kind; // "function" or "custom"
        std::string name;
        std::string ns;
        json        def;
    };
    std::vector<resp_tool>   resp_tools;
    std::vector<std::string> unsupported_tools;
    const bool has_tools = response_body.contains("tools");
    std::function<void(const json &, const std::string &)> add_resp_tool = [&](const json & tool, const std::string & ns) {
        if (!tool.is_object()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        const std::string type = json_value(tool, "type", std::string());
        if ((type == "function" || type == "custom") && tool.contains("name") && tool.at("name").is_string()) {
            resp_tools.push_back({type, tool.at("name").get<std::string>(), ns, tool});
        } else if (type == "namespace" && ns.empty() && tool.contains("tools") && tool.at("tools").is_array()) {
            const std::string name = json_value(tool, "name", std::string());
            for (const auto & child : tool.at("tools")) {
                add_resp_tool(child, name);
            }
        } else {
            const std::string what = type.empty() ? std::string("unknown") : type;
            if (std::find(unsupported_tools.begin(), unsupported_tools.end(), what) == unsupported_tools.end()) {
                unsupported_tools.push_back(what);
            }
        }
    };
    if (has_tools) {
        if (!response_body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        for (const auto & tool : response_body.at("tools")) {
            add_resp_tool(tool, "");
        }
    }
    bool has_additional_tools = false;
    if (input_value.is_array()) {
        for (const auto & item : input_value) {
            if (item.is_object() && json_value(item, "type", std::string()) == "additional_tools" &&
                    item.contains("tools") && item.at("tools").is_array()) {
                has_additional_tools = true;
                for (const auto & tool : item.at("tools")) {
                    add_resp_tool(tool, "");
                }
            }
        }
    }
    std::map<std::string, int> n_named;
    for (const auto & t : resp_tools) {
        n_named[t.name]++;
    }
    json chatcmpl_tools = json::array();
    json resp_tool_map  = json::object();
    std::map<std::string, std::string> model_name_of; // namespace + '\n' + name -> the name the model is shown
    for (const auto & t : resp_tools) {
        const std::string model_name = n_named[t.name] > 1 && !t.ns.empty() ? t.ns + "__" + t.name : t.name;
        model_name_of[t.ns + "\n" + t.name] = model_name;
        json fn;
        if (t.kind == "function") {
            fn = t.def;
            fn.erase("type");
            if (!fn.contains("strict")) {
                fn["strict"] = true;
            }
        } else {
            std::string input_desc = "The input for the tool as plain text, not JSON";
            const json  format     = json_value(t.def, "format", json::object());
            if (format.is_object() && json_value(format, "type", std::string()) == "grammar") {
                input_desc += ". It must follow this " + json_value(format, "syntax", std::string("lark")) +
                              " grammar:\n" + json_value(format, "definition", std::string());
            }
            fn = json {
                {"name",        t.name},
                {"description", json_value(t.def, "description", std::string())},
                {"parameters",  json {
                    {"type",       "object"},
                    {"properties", json { {"input", json { {"type", "string"}, {"description", input_desc} }} }},
                    {"required",   json::array({"input"})},
                }},
            };
        }
        fn["name"] = model_name;
        chatcmpl_tools.push_back(json { {"type", "function"}, {"function", fn} });
        if (t.kind == "custom" || !t.ns.empty() || model_name != t.name) {
            json entry = { {"type", t.kind}, {"name", t.name} };
            if (!t.ns.empty()) {
                entry["namespace"] = t.ns;
            }
            resp_tool_map[model_name] = entry;
        }
    }
    for (const auto & what : unsupported_tools) {
        SRV_WRN("Responses tool type '%s' is not supported by this server; the model is not offered it\n", what.c_str());
    }
    // the name the model knows a called tool by (a call in the input carries the client's name and namespace)
    auto model_tool_name = [&](const json & item) -> std::string {
        const std::string name = json_value(item, "name", std::string());
        const std::string ns   = json_value(item, "namespace", std::string());
        auto it = model_name_of.find(ns + "\n" + name);
        if (it != model_name_of.end()) {
            return it->second;
        }
        return name;
    };
    // a tool output as chat content: a string as it is, input_text parts as text parts, anything else described
    auto tool_output_content = [](const json & output) -> json {
        if (output.is_string()) {
            return output;
        }
        if (!output.is_array()) {
            return output.dump();
        }
        json parts = json::array();
        for (const auto & part : output) {
            const std::string type = json_value(part, "type", std::string());
            if ((type == "input_text" || type == "output_text" || type == "text") && part.contains("text") &&
                    part.at("text").is_string()) {
                parts.push_back(json { {"text", part.at("text")}, {"type", "text"} });
            } else if (type == "input_image" || type == "input_file") {
                SRV_WRN("a %s part of a tool output is not supported; the model is told it was left out\n", type.c_str());
                parts.push_back(json { {"text", "[" + type + " left out]"}, {"type", "text"} });
            } else {
                parts.push_back(json { {"text", part.dump()}, {"type", "text"} });
            }
        }
        return parts;
    };
    std::set<std::string> skipped_items;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            // strixllama: a custom tool call is a call of the function the custom tool became, its output that call's
            // output; additional_tools items only carry tools (collected above)
            if (exists_and_is_string(item, "type")) {
                const std::string item_type = item.at("type").get<std::string>();
                if (item_type == "additional_tools") {
                    continue;
                }
                if (item_type == "custom_tool_call") {
                    item["type"]      = "function_call";
                    item["arguments"] = json { {"input", json_value(item, "input", std::string())} }.dump();
                } else if (item_type == "custom_tool_call_output") {
                    item["type"] = "function_call_output";
                }
            }

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                // #responses_create-input-input_item_list-item-output_message
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(chatcmpl_content);
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (item.contains("arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                // strixllama: under the name the model was shown (namespace tools); arguments given as an object are
                // serialized
                const json & args = item.at("arguments");
                json tool_call = {
                    {"function", json {
                        {"arguments", args.is_string() ? args : json(args.dump())},
                        {"name",      model_tool_name(item)},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                item.contains("output") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                // strixllama: text parts as text, other parts described instead of failing the request
                chatcmpl_messages.push_back(json {
                    {"content",      tool_output_content(item.at("output"))},
                    {"role",         "tool"},
                    {"tool_call_id", item.at("call_id")},
                });
            } else if (exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning
                // strixllama: the reasoning text, else its summary; a reasoning item with neither (only
                // encrypted_content, as clients replay it) is left out instead of failing the request
                std::string text;
                if (exists_and_is_array(item, "content")) {
                    for (const auto & part : item.at("content")) {
                        if (exists_and_is_string(part, "text")) {
                            text += (text.empty() ? "" : "\n\n") + part.at("text").get<std::string>();
                        }
                    }
                }
                if (text.empty() && exists_and_is_array(item, "summary")) {
                    for (const auto & part : item.at("summary")) {
                        if (exists_and_is_string(part, "text")) {
                            text += (text.empty() ? "" : "\n\n") + part.at("text").get<std::string>();
                        }
                    }
                }
                if (text.empty()) {
                    continue;
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = text;
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", text},
                    });
                }
            } else if (exists_and_is_string(item, "type") && item.at("type") != "message") {
                // strixllama: items of hosted tools and other kinds this server has no use for (web_search_call,
                // local_shell_call, compaction, item_reference, ...) are left out instead of failing the request
                skipped_items.insert(item.at("type").get<std::string>());
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    chatcmpl_body["messages"] = chatcmpl_messages;
    for (const auto & what : skipped_items) {
        SRV_WRN("Responses input items of type '%s' are not supported by this server; left out\n", what.c_str());
    }

    if (has_tools || has_additional_tools) {
        chatcmpl_body.erase("tools");
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        }
        if (!resp_tool_map.empty()) {
            chatcmpl_body["strix_resp_tools"] = resp_tool_map;
        }
        if (!unsupported_tools.empty()) {
            chatcmpl_body["strix_resp_unsupported"] = unsupported_tools;
        }
    }

    // strixllama: a tool_choice object names one tool (function, custom) or a set (allowed_tools): the closest the
    // chat tool choice can say is "required" or the set's mode; a hosted tool choice is "auto"
    if (response_body.contains("tool_choice") && response_body.at("tool_choice").is_object()) {
        const json &      choice = response_body.at("tool_choice");
        const std::string type   = json_value(choice, "type", std::string());
        std::string       mapped = "auto";
        if (type == "function" || type == "custom") {
            mapped = "required";
        } else if (type == "allowed_tools") {
            mapped = json_value(choice, "mode", std::string("auto"));
        }
        chatcmpl_body["tool_choice"] = chatcmpl_tools.empty() ? std::string("auto") : mapped;
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    if (response_body.contains("reasoning")) {
        // Only "effort" is handled so far
        const json & reasoning = response_body.at("reasoning");
        if (reasoning.contains("effort")) {
            chatcmpl_body["reasoning_effort"] = reasoning.at("effort");
        }
        chatcmpl_body.erase("reasoning");
    }

    return chatcmpl_body;
}

// Edits the cch section of an "x-anthropic-billing-header" system prompt.
// Does nothing to any other prompt.
//
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint. It's not meaningful as a
// system prompt data, particularly to llama.cpp, but its presence means the prefix
// cache will not get past it: It changes on each request.
//
// Reference: https://github.com/ggml-org/llama.cpp/pull/21793
// Example header:
// ```
// x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, Anthropic's official CLI for Claude.
//                                                                            ^^^^^
// ```
static void normalize_anthropic_billing_header(std::string & system_text) {
    if (system_text.rfind("x-anthropic-billing-header:", 0) != 0) {
        return;
    }

    const size_t header_prefix_length = strlen("x-anthropic-billing-header:");
    const size_t cch_length = 5;
    const size_t index_cch = system_text.find("cch=", header_prefix_length);
    if (index_cch == std::string::npos) {
        return;
    }

    const size_t index_replace = index_cch + 4;
    if (index_replace + cch_length < system_text.length() && system_text[index_replace + cch_length] == ';') {
        for (size_t i = 0; i < cch_length; ++i) {
            system_text[index_replace + i] = 'f';
        }
    } else {
        LOG_ERR("anthropic string not as expected: %s", system_text.c_str());
    }
}

json server_chat_convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
            normalize_anthropic_billing_header(system_content);
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    auto system_text = json_value(block, "text", std::string());
                    normalize_anthropic_billing_header(system_text);
                    system_content += system_text;
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
        });
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            std::string reasoning_content;
            bool has_tool_calls = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    converted_content.push_back(block);
                } else if (type == "thinking") {
                    reasoning_content += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        });
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        });
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    if (result_content.is_string()) {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", result_content.get<std::string>()}
                        });
                    } else if (result_content.is_array()) {
                        // Single-pass: build both text and content_parts, decide format at the end
                        std::string result_text;
                        json content_parts = json::array();
                        bool has_images = false;

                        for (const auto & c : result_content) {
                            std::string c_type = json_value(c, "type", std::string());
                            if (c_type == "text") {
                                std::string text = json_value(c, "text", std::string());
                                result_text += text;
                                content_parts.push_back({
                                    {"type", "text"},
                                    {"text", text}
                                });
                            } else if (c_type == "image") {
                                has_images = true;
                                json source = json_value(c, "source", json::object());
                                std::string source_type = json_value(source, "type", std::string());
                                if (source_type == "base64") {
                                    std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                                    std::string data = json_value(source, "data", std::string());
                                    std::string url = "data:" + media_type + ";base64," + data;
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", url}}}
                                    });
                                } else if (source_type == "url") {
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", json_value(source, "url", std::string())}}}
                                    });
                                }
                            }
                        }

                        if (!has_images) {
                            // Text-only: collapse to a plain string for maximum compatibility
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", result_text}
                            });
                        } else {
                            // Mixed or image-only: use array content parts (OpenAI multimodal tool format)
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", content_parts}
                            });
                        }
                    } else {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", ""}
                        });
                    }
                }
            }

            if (!converted_content.empty() || has_tool_calls || !reasoning_content.empty()) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls || !reasoning_content.empty()) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                if (!reasoning_content.empty()) {
                    new_msg["reasoning_content"] = reasoning_content;
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    oai_body["messages"] = oai_messages;

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            for (const auto & tool : tools) {
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "any" || type == "tool") {
                oai_body["tool_choice"] = "required";
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        oai_body["max_tokens"] = body.at("max_tokens");
    } else {
        oai_body["max_tokens"] = 4096;
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        json thinking = json_value(body, "thinking", json::object());
        std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            oai_body["thinking_budget_tokens"] = budget_tokens;
        }
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    return oai_body;
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    if (response_format != "json") {
        throw std::invalid_argument("Only 'json' response_format is supported for transcription");
    }
    const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls);
    if (prompt.empty()) {
        prompt = preset.user;
    }
    if (!language.empty()) {
        prompt += string_format(" (language: %s)", language.c_str());
    }
    prompt += get_media_marker();

    json messages = json::array();
    if (!preset.system.empty()) {
        messages.push_back({{"role", "system"}, {"content", preset.system}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    if (inp_body.contains("temperature")) {
        std::string inp = inp_body["temperature"].get<std::string>();
        chatcmpl_body["temperature"] = std::stof(inp);
    }

    return chatcmpl_body;
}

// Tool-use checks: the model looks at its own agent turn before it goes out.
//
// For an OpenAI chat request that carries tools, the server
//   1. adds short working rules to the system message and turns thinking on,
//      unless the client chose a thinking setting itself;
//   2. re-plans a turn that calls two or more different tools at once: the
//      calls would run together, so the model is asked to make now only the
//      ones that need nothing from the others;
//   3. checks a final answer given after tool results, once per user request:
//      the model goes through the request part by part, and its new turn is
//      kept only if it makes tool calls that have not already failed.
//
// Every pass is the same model on the same cached conversation; there is no
// second model and nothing specific to one model family. The functions below
// only read and build OpenAI chat JSON, so they are tested without a model.

#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace luce::common::tool_checks {

using json = nlohmann::json;

// Thinking effort for a request that did not choose a thinking setting.
extern const char kEffort[];
extern const char kRules[];
extern const char kReplanNote[];
extern const char kFinalNote[];

// Tool calls added by a final check carry this id prefix. The ids travel with
// the conversation, so the next final answer of the same user request is not
// checked again and the server keeps no state between requests.
extern const char kCheckedCallPrefix[];

// The request gets checks: it has tools the model may call, the server has
// checks on, and the client did not set "tool_checks": false (top level or
// extra_body).
bool requested(const json & body, bool server_default);

// The request the model answers: rules added to the system message, thinking
// on at `effort` unless the client chose, non-streaming, and checks off so the
// inner request takes the plain path.
json agent_request(const json & body, const std::string & effort);

// `request` continued with the model's `reply` and a note from the server.
json with_note(const json & request, const json & reply, const char * note);

// The assistant message of a chat.completion response, or nullptr.
json * reply_message(json & response);

// The reply calls two or more different tools.
bool needs_replan(const json & reply);

// The reply is a final answer after tool results, does not end by asking the
// user something, and no check has added work since the last user message.
bool needs_final_check(const json & messages, const json & reply);

// The calls of `reply` that did not already fail earlier in `messages`.
json fresh_calls(const json & messages, const json & reply);

// Give each call the checked-call id prefix.
void mark_checked(json & calls);

// A chat.completion response as the SSE events a streaming client expects.
std::string as_sse(const json & response);

}  // namespace luce::common::tool_checks

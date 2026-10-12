// cluster_decision_hooks.cpp - head / worker / local decision hooks.

#include "cluster/cluster_decision_hooks.h"
#include "cluster/cluster_control.h"


namespace luce::cluster {

namespace {

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

}  // namespace

// ─── HeadHooks ─────────────────────────────────────────────────────────

HeadHooks::HeadHooks(ClusterHeadControl & control) : control_(control) {}

bool HeadHooks::decide_next_token(uint64_t req, uint32_t step, const float * /*logits*/,
                                  int /*n_vocab*/, const common::SamplerCfg & /*cfg*/,
                                  bool /*is_eos_candidate_unused*/,
                                  int32_t & token, uint8_t & flags, std::string * err) {
    // Rank 0 samples in its decode loop, with the request's sampler and RNG.
    if (token < 0) {
        set_err(err, "head: no token to broadcast");
        return false;
    }

    DecisionMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.token = token;
    msg.flags = flags;

    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.decisions++;
    return true;
}

bool HeadHooks::decide_draft(uint64_t req, uint32_t step, int pos, std::vector<int32_t> & tokens,
                             std::string * err) {
    DraftMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.pos = pos;
    msg.tokens = tokens;
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.drafts++;
    return true;
}

bool HeadHooks::decide_accept(uint64_t req, uint32_t step, int32_t & accept, int32_t & bonus,
                              uint8_t & flags, std::string * err) {
    AcceptMsg msg;
    msg.request_id = req;
    msg.step = step;
    msg.accept = accept;
    msg.bonus = bonus;
    msg.flags = flags;
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(msg), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.accepts++;
    return true;
}

bool HeadHooks::backend_op(const BackendOpMsg & op, std::string * err) {
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.broadcast(make_frame(op), err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    counters_.backend_ops++;
    return true;
}

// ─── WorkerHooks ───────────────────────────────────────────────────────

WorkerHooks::WorkerHooks(ClusterWorkerControl & control, const WorkerHooksConfig & cfg)
    : control_(control), cfg_(cfg) {}

bool WorkerHooks::take_pending_frame(Frame & out) {
    if (!has_pending_) return false;
    out = std::move(pending_);
    pending_ = Frame{};
    has_pending_ = false;
    return true;
}

bool WorkerHooks::recv_expected(MsgType want, Frame & frame, std::string * err) {
    if (aborted_) {
        set_err(err, "worker: head aborted: " + abort_reason_);
        return false;
    }
    if (has_pending_) {
        // A frame parked by an earlier mismatch is still unconsumed; the
        // main loop must handle it before the backend asks for decisions.
        set_err(err, std::string("worker: unconsumed ") + msg_type_name(pending_.type) +
                     " frame while waiting for " + msg_type_name(want));
        return false;
    }
    const uint64_t t0 = monotonic_us();
    const bool ok = control_.recv(frame, cfg_.timeout_ms, err);
    counters_.ctrl_wait_us += monotonic_us() - t0;
    if (!ok) return false;
    if (frame.type == want) return true;

    if (frame.type == MsgType::Abort) {
        AbortMsg abort;
        aborted_ = true;
        abort_reason_ = parse_frame(frame, abort) ? abort.reason : std::string("(malformed Abort)");
        set_err(err, "worker: head aborted: " + abort_reason_);
        return false;
    }
    pending_ = std::move(frame);
    has_pending_ = true;
    set_err(err, std::string("worker: expected ") + msg_type_name(want) + " but received " +
                 msg_type_name(pending_.type));
    return false;
}

bool WorkerHooks::decide_next_token(uint64_t req, uint32_t step, const float * /*logits*/,
                                    int /*n_vocab*/, const common::SamplerCfg & /*cfg*/,
                                    bool /*is_eos_candidate_unused*/, int32_t & token,
                                    uint8_t & flags, std::string * err) {
    Frame frame;
    if (!recv_expected(MsgType::Decision, frame, err)) return false;
    DecisionMsg msg;
    if (!parse_frame(frame, msg)) {
        set_err(err, "worker: malformed Decision frame");
        return false;
    }
    if (msg.request_id != req || msg.step != step) {
        set_err(err, "worker: Decision for req=" + std::to_string(msg.request_id) +
                     " step=" + std::to_string(msg.step) + ", expected req=" +
                     std::to_string(req) + " step=" + std::to_string(step));
        return false;
    }
    token = msg.token;
    flags = msg.flags;
    counters_.decisions++;
    return true;
}

bool WorkerHooks::decide_draft(uint64_t req, uint32_t step, int pos, std::vector<int32_t> & tokens,
                               std::string * err) {
    Frame frame;
    if (!recv_expected(MsgType::Draft, frame, err)) return false;
    DraftMsg msg;
    if (!parse_frame(frame, msg)) {
        set_err(err, "worker: malformed Draft frame");
        return false;
    }
    if (msg.request_id != req || msg.step != step) {
        set_err(err, "worker: Draft for req=" + std::to_string(msg.request_id) +
                     " step=" + std::to_string(msg.step) + ", expected req=" +
                     std::to_string(req) + " step=" + std::to_string(step));
        return false;
    }
    if (msg.pos != pos) {
        set_err(err, "worker: Draft pos=" + std::to_string(msg.pos) + " but local pos=" +
                     std::to_string(pos) + " (ranks diverged)");
        return false;
    }
    tokens = std::move(msg.tokens);
    counters_.drafts++;
    return true;
}

bool WorkerHooks::decide_accept(uint64_t req, uint32_t step, int32_t & accept, int32_t & bonus,
                                uint8_t & flags, std::string * err) {
    Frame frame;
    if (!recv_expected(MsgType::Accept, frame, err)) return false;
    AcceptMsg msg;
    if (!parse_frame(frame, msg)) {
        set_err(err, "worker: malformed Accept frame");
        return false;
    }
    if (msg.request_id != req || msg.step != step) {
        set_err(err, "worker: Accept for req=" + std::to_string(msg.request_id) +
                     " step=" + std::to_string(msg.step) + ", expected req=" +
                     std::to_string(req) + " step=" + std::to_string(step));
        return false;
    }
    accept = msg.accept;
    bonus = msg.bonus;
    flags = msg.flags;
    counters_.accepts++;
    return true;
}

}  // namespace luce::cluster

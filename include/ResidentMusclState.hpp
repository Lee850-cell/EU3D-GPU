#pragma once

struct ReactionFlowState;

// Host-side view only. The allocation is owned by the EE state bridge.
// Published after reaction, consumed by MUSCL before the next EE update.
struct ResidentMusclState {
    bool full = false; // Single-rank GPU boundaries and deferred host scatter.
    bool shadow = false; // Keep host state current for boundary comparison.
    bool dirty = false;
    const double *ri = nullptr, *coeff0 = nullptr, *coeff1 = nullptr;
    const double* sources = nullptr;
    const int* nchem = nullptr;
    int max_nchem = 0;
    bool enabled = false;
    bool ready = false;
    bool validate = false;
    const ReactionFlowState* states = nullptr;
    int ni = 0, nj = 0, nk = 0, bc = 0, ns = 0;
};

inline ResidentMusclState resident_muscl_state;

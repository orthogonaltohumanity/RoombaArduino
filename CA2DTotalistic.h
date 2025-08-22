#ifndef CA2D_TOTALISTIC_H
#define CA2D_TOTALISTIC_H

#include <Arduino.h>

/*
  CA2DTotalistic.h
  -----------------
  Minimal 2D binary totalistic cellular automaton with Moore neighborhood.

  - Grid: 32x32 torus (1024 cells -> 1024 bits -> 128 bytes per buffer)
  - Rule: 18-bit table indexed by (current_state in {0,1}, neighbor_count in 0..8)
          bit index = current_state*9 + neighbor_count
  - Update: synchronous by default; optional sparse/async via 16-bit LFSR mask (update_prob 0..255)
  - Seed: density-controlled random fill via same LFSR
  - Mapping: Hilbert-curve order → packed bytes for NN weights (LSB-first)
  - No dynamic memory; two buffers (grid,next) = 256 bytes total.

  Usage (example):
    CA2DTotalistic ca;
    ca.setRule18(0b000000111_001110000u); // example rule bits (D/C & neighbor-count order)
    ca.seedRandom(0xACE1u, 16);           // density ≈ 16/32 = 0.5
    ca.run(48, 64);                        // 48 steps, ~25% of cells updated per step
    // Map to your net weights (nBits <= 1024):
    ca.mapHilbertToBytes(out_weight_bytes, nBits, /*invert=*/false);

  Notes:
    - If nBits < 1024, only the first nBits along the Hilbert path are emitted.
    - If nBits > 1024, extra bits are zeroed (you can call map twice with different seeds if needed).
    - Bit packing is LSB-first per byte: bit i goes to (out[i>>3] >> (i&7)).
*/

struct CA2DTotalistic {
  static constexpr uint8_t W = 32;
  static constexpr uint8_t H = 32;
  static constexpr uint16_t N = W * H;
  static constexpr uint16_t BYTES = (N + 7) >> 3;

  // Bit-packed grids
  uint8_t grid[BYTES];
  uint8_t next[BYTES];

  // 18-bit rule: index = (cur * 9 + neighborCount), value = next_state bit
  // Stored in the lower 18 bits of rule18.
  uint32_t rule18;

  // 16-bit LFSR for RNG (Xorshift-like taps)
  uint16_t lfsr = 0xBEEF;

  CA2DTotalistic() : rule18(0) { clear(grid); clear(next); }

  // ---- Basic helpers ----
  static inline void clear(uint8_t* b) {
    for (uint16_t i = 0; i < BYTES; ++i) b[i] = 0;
  }

  static inline bool getBit(const uint8_t* b, uint16_t idx) {
    return (b[idx >> 3] >> (idx & 7)) & 1;
  }
  static inline void setBit(uint8_t* b, uint16_t idx, bool v) {
    uint8_t mask = (uint8_t)1 << (idx & 7);
    if (v) b[idx >> 3] |= mask; else b[idx >> 3] &= ~mask;
  }

  inline uint16_t rnd16() {
    // 16-bit Galois LFSR (tap 0xB400 recommended; using 0xB400u)
    uint16_t l = lfsr;
    uint16_t lsb = l & 1u;
    l >>= 1;
    if (lsb) l ^= 0xB400u;
    lfsr = l;
    return l;
  }

  inline uint8_t rnd8() { return (uint8_t)(rnd16() & 0xFF); }

  // ---- Rule setup ----
  inline void setRule18(uint32_t rule) {
    rule18 = (rule & 0x3FFFFu); // keep 18 bits
  }

  // Convenience: set rule from two 9-bit masks:
  // deadTable: bits for next_state when current cell = 0, indexed by neighbor count 0..8
  // liveTable: bits for next_state when current cell = 1, indexed by neighbor count 0..8
  inline void setRuleFromTables(uint16_t deadTable, uint16_t liveTable) {
    // Only low 9 bits of each are used
    rule18 = ((uint32_t)(deadTable & 0x1FFu)) | ((uint32_t)(liveTable & 0x1FFu) << 9);
  }

  // Example: Conway's Life would be deadTable=bit(3), liveTable=bits(2,3)
  // deadTable = 1<<3; liveTable = (1<<2) | (1<<3);

  // ---- Seeding ----
  // density_q5 in [0..31] → target alive probability ≈ density_q5/32
  inline void seedRandom(uint16_t seed, uint8_t density_q5) {
    lfsr = seed ? seed : 1;
    clear(grid);
    const uint8_t thr = density_q5; // compare rnd5 to thr
    for (uint16_t i = 0; i < N; ++i) {
      uint8_t r = rnd8() & 31; // 5-bit random
      setBit(grid, i, (r < thr));
    }
  }

  // Optional: sprinkle K motifs (tiny plus-shapes) for structured seeds
  inline void sprinkleMotif(uint8_t x, uint8_t y) {
    auto idx = [&](int8_t xx, int8_t yy)->uint16_t {
      uint8_t X = (uint8_t)((xx + W) & (W - 1));
      uint8_t Y = (uint8_t)((yy + H) & (H - 1));
      return (uint16_t)Y * W + X;
    };
    setBit(grid, idx(x, y), true);
    setBit(grid, idx(x+1, y), true);
    setBit(grid, idx(x-1, y), true);
    setBit(grid, idx(x, y+1), true);
    setBit(grid, idx(x, y-1), true);
  }

  // ---- Evolution ----
  // Count Moore neighbors on a torus
  inline uint8_t neighborCount(const uint8_t* b, uint8_t x, uint8_t y) const {
    uint8_t cnt = 0;
    for (int8_t dy = -1; dy <= 1; ++dy) {
      uint8_t Y = (uint8_t)((y + dy + H) & (H - 1));
      for (int8_t dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        uint8_t X = (uint8_t)((x + dx + W) & (W - 1));
        uint16_t idx = (uint16_t)Y * W + X;
        cnt += getBit(b, idx);
      }
    }
    return cnt; // 0..8
  }

  // One CA step.
  // update_prob in [0..255]: chance (per cell) that it *updates* this step. If not, it keeps old state (async sparsity).
  inline void step(uint8_t update_prob = 255) {
    for (uint8_t y = 0; y < H; ++y) {
      for (uint8_t x = 0; x < W; ++x) {
        uint16_t idx = (uint16_t)y * W + x;
        bool cur = getBit(grid, idx);
        bool nxt = cur;
        if (update_prob == 255 || rnd8() <= update_prob) {
          uint8_t nc = neighborCount(grid, x, y);
          uint8_t ruleIdx = (cur ? 9 : 0) + nc; // 0..17
          nxt = ((rule18 >> ruleIdx) & 1u);
        }
        setBit(next, idx, nxt);
      }
    }
    // swap buffers
    for (uint16_t i = 0; i < BYTES; ++i) grid[i] = next[i];
  }

  inline void run(uint8_t T, uint8_t update_prob = 255) {
    for (uint8_t t = 0; t < T; ++t) step(update_prob);
  }

  // ---- Mapping to weights ----
  // Hilbert curve helpers (for power-of-two square size)
  static inline void rot(uint8_t n, uint8_t &x, uint8_t &y, uint8_t rx, uint8_t ry) {
    if (ry == 0) {
      if (rx == 1) { x = n - 1 - x; y = n - 1 - y; }
      uint8_t t = x; x = y; y = t;
    }
  }

  // Convert Hilbert distance d -> (x,y) for n=32
  static inline void hilbert_d2xy(uint8_t n, uint16_t d, uint8_t &x, uint8_t &y) {
    uint16_t t = d;
    x = y = 0;
    for (uint8_t s = 1; s < n; s <<= 1) {
      uint8_t rx = (uint8_t)((t >> 1) & 1u);
      uint8_t ry = (uint8_t)((t ^ rx) & 1u);
      rot(s, x, y, rx, ry);
      x += s * rx;
      y += s * ry;
      t >>= 2;
    }
  }

  // Map first nBits along Hilbert path to outBytes (LSB-first). If invert=true, invert bits.
  inline void mapHilbertToBytes(uint8_t* outBytes, uint16_t nBits, bool invert=false) const {
    uint16_t outBytesN = (nBits + 7) >> 3;
    for (uint16_t i = 0; i < outBytesN; ++i) outBytes[i] = 0; // clear
    uint16_t limit = (nBits <= N) ? nBits : N;

    for (uint16_t i = 0; i < limit; ++i) {
      uint8_t x, y;
      hilbert_d2xy(W, i, x, y);
      uint16_t idx = (uint16_t)y * W + x;
      bool bit = getBit(grid, idx);
      if (invert) bit = !bit;

      uint16_t ob = i >> 3;
      uint8_t  off = i & 7;
      if (bit) outBytes[ob] |= (uint8_t)1 << off;
    }
    // If nBits > N, leave remaining bits zero.
  }

  // Simple alternative: row-major mapping (faster, less locality-preserving)
  inline void mapRowMajorToBytes(uint8_t* outBytes, uint16_t nBits, bool invert=false) const {
    uint16_t outBytesN = (nBits + 7) >> 3;
    for (uint16_t i = 0; i < outBytesN; ++i) outBytes[i] = 0;
    uint16_t limit = (nBits <= N) ? nBits : N;

    for (uint16_t i = 0; i < limit; ++i) {
      bool bit = getBit(grid, i);
      if (invert) bit = !bit;
      uint16_t ob = i >> 3;
      uint8_t  off = i & 7;
      if (bit) outBytes[ob] |= (uint8_t)1 << off;
    }
  }
};

#endif // CA2D_TOTALISTIC_H


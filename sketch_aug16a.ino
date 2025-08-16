#include <Arduino.h>
#include <string.h>

// Forward declaration so auto-generated prototypes can reference Layer
struct Layer;


/*
  Bit-Packed Binary Layer + Compact Genetic Algorithm (cGA)
  --------------------------------------------------------
  This sketch evolves a simple bit-packed weight matrix with a compact
  genetic algorithm to imitate a fixed random teacher.
*/

// ======== Dimensions ========
#define INPUT_DIM_BITS   32
#define OUTPUT_DIM        8

#define BYTES_PER_ROW   ((INPUT_DIM_BITS + 7) / 8)
#define OUT_BYTES       ((OUTPUT_DIM + 7) / 8)
#define NUM_BITS        ((uint32_t)OUTPUT_DIM * (uint32_t)INPUT_DIM_BITS)

#define BATCH_SIZE       16
#define CGA_N            50
#define CGA_DELTA         1
#define PRINT_EVERY      50

// ======== Utilities ========
static inline uint8_t popcount8(uint8_t v) { return (uint8_t)__builtin_popcount((unsigned int)v); }

static inline void setPackedBit(uint8_t *arr, uint32_t idx, bool v) {
  uint32_t byteIdx = idx >> 3;
  uint8_t  bitIdx  = idx & 0x07;
  uint8_t  mask    = (uint8_t)1 << bitIdx;
  if (v) arr[byteIdx] |= mask; else arr[byteIdx] &= (uint8_t)~mask;
}

static inline uint8_t getPackedBit(const uint8_t *arr, uint32_t idx) {
  uint32_t byteIdx = idx >> 3;
  uint8_t  bitIdx  = idx & 0x07;
  return (arr[byteIdx] >> bitIdx) & 0x01;
}

// ======== Bit-Matrix Layer ========
struct Layer {
  uint8_t weights[OUTPUT_DIM][BYTES_PER_ROW];

  void clear() { memset(weights, 0, sizeof(weights)); }

  void randomize() {
    for (uint8_t r = 0; r < OUTPUT_DIM; r++) {
      for (uint8_t b = 0; b < BYTES_PER_ROW; b++) {
        weights[r][b] = (uint8_t)random(0, 256);
      }
      uint8_t extraBits = (uint8_t)(BYTES_PER_ROW * 8 - INPUT_DIM_BITS);
      if (extraBits) {
        uint8_t mask = (uint8_t)(0xFF >> extraBits);
        weights[r][BYTES_PER_ROW - 1] &= mask;
      }
    }
  }

  void setBit(uint8_t row, uint16_t col, bool v) {
    uint16_t byteIdx = col >> 3;
    uint8_t  bitIdx  = col & 0x07;
    uint8_t  mask    = (uint8_t)1 << bitIdx;
    if (v) weights[row][byteIdx] |= mask; else weights[row][byteIdx] &= (uint8_t)~mask;
  }

  uint8_t getBit(uint8_t row, uint16_t col) const {
    uint16_t byteIdx = col >> 3;
    uint8_t  bitIdx  = col & 0x07;
    return (weights[row][byteIdx] >> bitIdx) & 0x01;
  }

  void forwardParity(const uint8_t *inPacked, uint8_t *outPacked) const {
    for (uint8_t i = 0; i < OUT_BYTES; i++) outPacked[i] = 0;
    for (uint8_t r = 0; r < OUTPUT_DIM; r++) {
      uint16_t acc = 0;
      for (uint8_t b = 0; b < BYTES_PER_ROW; b++) acc += popcount8((uint8_t)(weights[r][b] & inPacked[b]));
      uint8_t parity = (uint8_t)(acc & 1);
      if (parity) outPacked[r >> 3] |= (uint8_t)1 << (r & 0x07);
    }
  }
};

// ======== Globals ========
static Layer target, candA, candB, approx;
static uint8_t X[BATCH_SIZE][BYTES_PER_ROW];
static uint8_t Yt[BATCH_SIZE][OUT_BYTES];
static uint8_t p[NUM_BITS];
static uint8_t gA[(NUM_BITS + 7) / 8];
static uint8_t gB[(NUM_BITS + 7) / 8];

// ======== Helpers ========
static void genomeToLayer(const uint8_t *gPacked, Layer &out) {
  out.clear();
  uint32_t idx = 0;
  for (uint8_t r = 0; r < OUTPUT_DIM; r++) {
    for (uint16_t c = 0; c < INPUT_DIM_BITS; c++, idx++) {
      if (getPackedBit(gPacked, idx)) out.setBit(r, c, true);
    }
  }
}

static void pToLayerThreshold(const uint8_t *prob, Layer &out, uint8_t threshold) {
  out.clear();
  uint32_t idx = 0;
  for (uint8_t r = 0; r < OUTPUT_DIM; r++) {
    for (uint16_t c = 0; c < INPUT_DIM_BITS; c++, idx++) {
      if (prob[idx] >= threshold) out.setBit(r, c, true);
    }
  }
}

static void sampleGenome(const uint8_t *prob, uint8_t *gPacked) {
  memset(gPacked, 0, (NUM_BITS + 7) / 8);
  for (uint32_t i = 0; i < NUM_BITS; i++) {
    uint8_t r = (uint8_t)random(0, CGA_N);
    bool bit1 = (r < prob[i]);
    setPackedBit(gPacked, i, bit1);
  }
}

static void regenerateBatch() {
  for (uint8_t n = 0; n < BATCH_SIZE; n++) {
    for (uint8_t b = 0; b < BYTES_PER_ROW; b++) X[n][b] = (uint8_t)random(0, 256);
    uint8_t extraBits = (uint8_t)(BYTES_PER_ROW * 8 - INPUT_DIM_BITS);
    if (extraBits) X[n][BYTES_PER_ROW - 1] &= (uint8_t)(0xFF >> extraBits);
    target.forwardParity(X[n], Yt[n]);
  }
}

static uint16_t fitnessOnBatch(const Layer &L) {
  uint16_t fit = 0;
  uint8_t y[OUT_BYTES];
  for (uint8_t n = 0; n < BATCH_SIZE; n++) {
    L.forwardParity(X[n], y);
    for (uint8_t i = 0; i < OUT_BYTES; i++) {
      uint8_t same = (uint8_t)~(y[i] ^ Yt[n][i]);
      fit += popcount8(same);
    }
  }
  return fit;
}

static void initCGA() {
  for (uint32_t i = 0; i < NUM_BITS; i++) p[i] = (uint8_t)(CGA_N / 2);
  memset(gA, 0, sizeof(gA));
  memset(gB, 0, sizeof(gB));
}

// ======== Diagnostics ========
static void printByteBits(uint8_t v) {
  for (int8_t b = 7; b >= 0; b--) Serial.print((v >> b) & 1);
}

static void printLayerRow(const Layer &L, uint8_t r) {
  Serial.print(F("W[")); Serial.print(r); Serial.print(F("] "));
  for (uint8_t b = 0; b < BYTES_PER_ROW; b++) {
    printByteBits(L.weights[r][b]); Serial.print(' ');
  }
  Serial.println();
}

// ======== Arduino Setup/Loop ========
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  randomSeed(analogRead(A0));

  Serial.println(F("[cGA] Booting..."));
  target.randomize();
  initCGA();
  regenerateBatch();
  Serial.println(F("Teacher sample rows:"));
  for (uint8_t r = 0; r < min((uint8_t)2, (uint8_t)OUTPUT_DIM); r++) printLayerRow(target, r);
}

void loop() {
  static uint32_t iter = 0;
  sampleGenome(p, gA);
  sampleGenome(p, gB);
  genomeToLayer(gA, candA);
  genomeToLayer(gB, candB);

  uint16_t fA = fitnessOnBatch(candA);
  uint16_t fB = fitnessOnBatch(candB);

  if (fA != fB) {
    const uint8_t *win = (fA > fB) ? gA : gB;
    const uint8_t *los = (fA > fB) ? gB : gA;
    for (uint32_t i = 0; i < NUM_BITS; i++) {
      uint8_t a = getPackedBit(win, i);
      uint8_t b = getPackedBit(los, i);
      if (a != b) {
        if (a && p[i] + CGA_DELTA <= CGA_N) p[i] += CGA_DELTA;
        else if (!a && p[i] >= CGA_DELTA) p[i] -= CGA_DELTA;
      }
    }
  }

  iter++;
  if ((iter % (PRINT_EVERY * 2)) == 0) regenerateBatch();

  if ((iter % PRINT_EVERY) == 0) {
    pToLayerThreshold(p, approx, (uint8_t)(CGA_N / 2));
    uint16_t fApprox = fitnessOnBatch(approx);
    Serial.print(F("it=")); Serial.print(iter);
    Serial.print(F(" f*=")); Serial.print(fApprox);
    Serial.print(F(" / max=")); Serial.print((uint16_t)BATCH_SIZE * OUTPUT_DIM);
    Serial.println();
  }
}

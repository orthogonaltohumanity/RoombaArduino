#include <Arduino.h>
#include <Servo.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <SPI.h>
#include <SD.h>


const uint8_t FY_MAX_SWAPS = 4;     // keep your current value if already defined
// Total capacity of the swap log per trial (for 4 layers)
const uint8_t SWAP_CAP      = FY_MAX_SWAPS * 4;  // adjust if MAX_LAYERS != 4
// -----------------------------
// Hardware pins (same as other sketches)
// -----------------------------


#define LDR_L_PIN   A4
#define LDR_R_PIN   A5
#define LDR_C_PIN   A3
#define BUTTON_PIN  8
#define PLATE_PIN   9
#define SERVO_PIN   A0
#define BEEP_PIN    A1
#define ENA         6
#define IN1         2
#define IN2         3
#define ENB         5
#define IN3         4
#define IN4         7
#define SD_CS_PIN   10

// Fixed PWM speed for "on" states
const uint8_t MOTOR_PWM_ON = 255;

// SD card filenames for neural network weights
const char *ACT_NET_FILE = "actnet.bin";
const char *EMB_NET_FILE = "embnet.bin";
bool sd_ready = false;

// Map 9 bins to (L_dir, R_dir): -1=backward, 0=off, +1=forward
const int8_t MOTOR9_DIRS[9][2] = {
  {-1, -1},  // 0: L back,  R back
  {-1,  0},  // 1: L back,  R off
  {-1, +1},  // 2: L back,  R fwd
  { 0, -1},  // 3: L off,   R back
  { 0,  0},  // 4: L off,   R off  (stop)
  { 0, +1},  // 5: L off,   R fwd
  {+1, -1},  // 6: L fwd,   R back
  {+1,  0},  // 7: L fwd,   R off
  {+1, +1}   // 8: L fwd,   R fwd
};

// -----------------------------
// Problem sizes
// -----------------------------
const uint8_t N_MOTOR = 9;
const uint8_t N_SERVO = 10;
const uint8_t N_BEEP  = 5;
const uint8_t ACT_DIM = N_MOTOR + N_SERVO + N_BEEP; // 24
const uint8_t EMB_DIM = 8;              // embedding size (multiple of 8)
const uint8_t FEAT_DIM = 16;            // [ldrL, ldrR, ldrC, plate, button, last action embedding (8), bias, padding]
const uint8_t IDX_MOTOR0 = 0;
const uint8_t IDX_SERVO0 = IDX_MOTOR0 + N_MOTOR;   // 9
const uint8_t IDX_BEEP0  = IDX_SERVO0 + N_SERVO;   // 19

// -----------------------------
// Reward parameters
// -----------------------------
const int16_t BUTTON_REWARD_Q8  = 256;  // reward when button is pressed
const int16_t PLATE_PENALTY_Q8 = -256; // penalty when collision plates touch
const int16_t MOTOR_IDLE_PENALTY_Q8 = -64; // penalty when both motors stopped

// -----------------------------
// Beep patterns
// -----------------------------
const uint8_t BEEP_TONES = 5;
const uint16_t BEEP_FREQ[N_BEEP][BEEP_TONES] = {
  {262, 330, 392, 523, 659},
  {659, 523, 392, 330, 262},
  {262, 392, 523, 392, 262},
  {330, 262, 330, 392, 523},
  {523, 440, 392, 440, 523}
};
const uint16_t BEEP_DUR[N_BEEP][BEEP_TONES] = {
  {50, 50, 50, 50, 100},
  {100, 50, 50, 50, 100},
  {75, 75, 150, 75, 75},
  {50, 100, 50, 100, 150},
  {150, 75, 75, 75, 150}
};

// -----------------------------
// Helper structs
// -----------------------------
struct Feat {
  uint8_t x[FEAT_DIM];
};

uint16_t urand16();
struct HCState {
  bool active;
  int32_t reward_accum;
  int32_t reward_avg;
  uint16_t steps;
  uint8_t swap_count;
  uint16_t swap_from[SWAP_CAP];  // was [4]
  uint16_t swap_to[SWAP_CAP];    // was [4]
};

// -----------------------------
// Binary Feedforward Neural Network
// -----------------------------
class BinaryNN {
public:
  static const uint8_t MAX_LAYERS = 4;

  struct Layer {
    uint8_t in_dim;
    uint8_t out_dim;
    uint16_t w_off;  // bit offset for weights
  };

  Layer layers[MAX_LAYERS];
  uint8_t layer_count = 0;
  uint16_t total_bits = 0;

  uint8_t *weight_bytes = nullptr;
  uint8_t *acts = nullptr;
  int16_t *bufA = nullptr;
  int16_t *bufB = nullptr;
  uint8_t max_nodes = 0;

  void addLayer(uint8_t in_dim, uint8_t out_dim) {
    if (layer_count >= MAX_LAYERS) return;
    if ((in_dim & 7) || (out_dim & 7)) return; // require multiples of 8
    Layer &L = layers[layer_count];
    L.in_dim = in_dim;
    L.out_dim = out_dim;
    L.w_off = total_bits;
    total_bits += (uint16_t)in_dim * out_dim;
    layer_count++;
  }

  void initBuffers() {
    uint16_t nB = (total_bits + 7) >> 3;
    weight_bytes = (uint8_t*)malloc(nB);
    max_nodes = 0;
    for (uint8_t l=0; l<layer_count; ++l) {
      if (layers[l].in_dim > max_nodes) max_nodes = layers[l].in_dim;
      if (layers[l].out_dim > max_nodes) max_nodes = layers[l].out_dim;
    }
    acts = (uint8_t*)malloc((layer_count + 1) * max_nodes);
    bufA = (int16_t*)malloc(max_nodes * sizeof(int16_t));
    bufB = (int16_t*)malloc(max_nodes * sizeof(int16_t));
  }

  uint16_t numWeightBits() const { return total_bits; }

  void randomizeWeights() {
    uint16_t nB = (total_bits + 7) >> 3;
    for (uint16_t i=0;i<nB;++i) {
      weight_bytes[i] = (uint8_t)(urand16() & 0xFF);
    }
  }

  inline int8_t getWeight(uint16_t bit_index) const {
    return (weight_bytes[bit_index>>3] >> (bit_index & 7)) & 1 ? 1 : -1;
  }

  inline uint8_t* layerActs(uint8_t layer_idx) const {
    return acts + layer_idx * max_nodes;
  }

  void forward(const uint8_t *input, int16_t *out) {
    uint8_t *acts0 = layerActs(0);
    for (uint8_t i=0;i<layers[0].in_dim;++i) acts0[i] = (input[i] > 127) ? 1 : 0;
    for (uint8_t i=0;i<layers[0].in_dim;++i) bufA[i] = input[i];
    int16_t *cur_in = bufA;
    int16_t *cur_out = bufB;
    for (uint8_t l=0;l<layer_count;++l) {
      Layer &L = layers[l];
      uint8_t in_dim = L.in_dim;
      uint8_t out_dim = L.out_dim;
      uint16_t off = L.w_off;
      int16_t *next_out = (l==layer_count-1) ? out : cur_out;
      uint8_t *acts_next = layerActs(l+1);
      for (uint8_t j=0;j<out_dim;++j) {
        int16_t sum = 0;
        for (uint8_t i=0;i<in_dim;++i) {
          int8_t w = getWeight(off + j*in_dim + i);
          sum += w * cur_in[i];
        }
        if (l==layer_count-1) {
          next_out[j] = sum;
          acts_next[j] = (sum >= 0) ? 1 : 0;
        } else {
          int16_t act = (sum >= 0 ? 1 : -1);
          next_out[j] = act;
          acts_next[j] = (act > 0) ? 1 : 0;
        }
      }
      if (l != layer_count-1) {
        int16_t *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
      }
    }
  }

  void toggleWeight(uint16_t bit_index) {
    uint8_t mask = (uint8_t)1 << (bit_index & 7);
    weight_bytes[bit_index>>3] ^= mask;
  }

  void swapWeights(uint16_t a, uint16_t b) {
    uint8_t wa = (weight_bytes[a>>3] >> (a & 7)) & 1;
    uint8_t wb = (weight_bytes[b>>3] >> (b & 7)) & 1;
    if (wa != wb) {
      toggleWeight(a);
      toggleWeight(b);
    }
  }

  ~BinaryNN() {
    free(weight_bytes);
    free(acts);
    free(bufA);
    free(bufB);
  }
};

BinaryNN act_net, embed_net;
// Round-robin layer pointers for each net
uint8_t rr_layer_act = 0;
uint8_t rr_layer_emb = 0;

bool saveNetwork(const BinaryNN &net, const char *filename) {
  uint16_t nB = (net.numWeightBits() + 7) >> 3;
  SD.remove(filename);
  File f = SD.open(filename, FILE_WRITE);
  if (!f) {
    Serial.print("Failed to open "); Serial.println(filename);
    return false;
  }
  size_t written = f.write(net.weight_bytes, nB);
  f.close();
  if (written != nB) {
    Serial.print("Short write "); Serial.println(filename);
    return false;
  }
  Serial.print("Saved "); Serial.println(filename);
  return true;
}

bool loadNetwork(BinaryNN &net, const char *filename) {
  File f = SD.open(filename, FILE_READ);
  if (!f) {
    Serial.print("Missing "); Serial.println(filename);
    return false;
  }
  uint16_t nB = (net.numWeightBits() + 7) >> 3;
  if (f.size() < nB) {
    f.close();
    Serial.print("Size mismatch "); Serial.println(filename);
    return false;
  }
  size_t rd = f.read(net.weight_bytes, nB);
  f.close();
  if (rd != nB) {
    Serial.print("Short read "); Serial.println(filename);
    return false;
  }
  Serial.print("Loaded "); Serial.println(filename);
  return true;
}

const uint8_t LAMBDA_MAX = 8;   // cap; tune as you like
const uint8_t HC_STEPS   = 50;  // keep your existing value if already defined
// How many swaps per layer to attempt


struct AxisSelector {
  // Single lambda controls locality and Hebbian similarity bias.
  uint8_t lambda;
};





AxisSelector sel_act, sel_emb;
HCState hc_act = {false,0,0,0,0,{0},{0}};
HCState hc_emb = {false,0,0,0,0,{0},{0}};
const uint8_t HC_AVG_SHIFT = 3; // 1/8 smoothing

void initAxisSelector(AxisSelector &sel, uint16_t /*n*/) {
  // Start moderately local/similar. You can start flatter by using 0..2.
  sel.lambda = 3;
}

// Sample a swap distance using a truncated geometric law. Let s = 2^-lambda.
// For 0 <= k < D:  Pr[d=k] = s^k (1 - s);  Pr[d=D] = s^D.
// Larger lambda -> smaller s -> stronger bias toward short hops.
// Integer-only; no divisions; uses lambda low-bit mask checks
inline uint16_t sample_distance_geometric(uint16_t D, uint8_t lambda) {
  if (D == 0) return 0;
  if (lambda == 0) {           // Uniform over [0..D]
    return urand16() % (D + 1);
  }
  uint16_t d = 0;
  // mask of lambda bits: if lambda >=16, success prob is tiny; clamp
  uint16_t mask = (lambda >= 16) ? 0xFFFF : ((1U << lambda) - 1U);
  while (d < D) {
    uint16_t r = urand16();
    // "success" means the lowest lambda bits are all zero (prob = 2^-lambda)
    if ((r & mask) == 0) {
      ++d;                     // keep stepping locally
    } else {
      break;                   // first failure -> stop
    }
  }
  return d;
}


// Return [lo, hi] (inclusive) bit-index range for layer L
inline void getLayerBitRange(const BinaryNN &net, uint8_t L, uint16_t &lo, uint16_t &hi){
  const uint16_t off  = net.layers[L].w_off;
  const uint16_t size = (uint16_t)net.layers[L].in_dim * net.layers[L].out_dim;
  lo = off;
  hi = off + size - 1;
}


float cosine_rows(const BinaryNN &net, uint8_t L, uint8_t r1, uint8_t r2){
  const uint16_t off = net.layers[L].w_off;
  const uint8_t in_dim = net.layers[L].in_dim;
  const uint8_t out_dim = net.layers[L].out_dim;
  int16_t dot=0;
  for(uint8_t c=0;c<out_dim;c++){
    int8_t w1 = net.getWeight(off + c*in_dim + r1);
    int8_t w2 = net.getWeight(off + c*in_dim + r2);
    dot += w1*w2;
  }
  return (float)dot / out_dim;
}

float cosine_cols(const BinaryNN &net, uint8_t L, uint8_t c1, uint8_t c2){
  const uint16_t off = net.layers[L].w_off;
  const uint8_t in_dim = net.layers[L].in_dim;
  const uint8_t out_dim = net.layers[L].out_dim;
  int16_t dot=0;
  for(uint8_t r=0;r<in_dim;r++){
    int8_t w1 = net.getWeight(off + c1*in_dim + r);
    int8_t w2 = net.getWeight(off + c2*in_dim + r);
    dot += w1*w2;
  }
  return (float)dot / in_dim;
}

float cosine_col_row(const BinaryNN &net, uint8_t L, uint8_t c, uint8_t r){
  const uint16_t off = net.layers[L].w_off;
  const uint8_t in_dim = net.layers[L].in_dim;
  const uint8_t out_dim = net.layers[L].out_dim;
  uint8_t Lmin = in_dim < out_dim ? in_dim : out_dim;
  int16_t dot=0;
  for(uint8_t t=0;t<Lmin;t++){
    int8_t w_col = net.getWeight(off + c*in_dim + t);
    int8_t w_row = net.getWeight(off + t*in_dim + r);
    dot += w_col * w_row;
  }
  return (float)dot / Lmin;
}

float cosine_row_col(const BinaryNN &net, uint8_t L, uint8_t r, uint8_t c){
  return cosine_col_row(net, L, c, r);
}

uint16_t selectSwapPartner(const BinaryNN &net, uint8_t L, uint16_t idx_i, const AxisSelector &sel){
  const BinaryNN::Layer &layer = net.layers[L];
  const uint16_t off = layer.w_off;
  const uint8_t in_dim = layer.in_dim;
  const uint8_t out_dim = layer.out_dim;
  uint16_t rel = idx_i - off;
  uint8_t r1 = rel % in_dim;
  uint8_t c1 = rel / in_dim;
  const float w_max = powf(2.0f, 4 * sel.lambda);
  for(uint8_t attempt=0; attempt<8; ++attempt){
    uint16_t dr = sample_distance_geometric(in_dim-1, sel.lambda);
    uint16_t dc = sample_distance_geometric(out_dim-1, sel.lambda);
    int8_t sr = (urand16() & 1) ? 1 : -1;
    int8_t sc = (urand16() & 1) ? 1 : -1;
    int16_t r2 = (int16_t)r1 + sr*dr;
    int16_t c2 = (int16_t)c1 + sc*dc;
    if(r2<0 || r2>=in_dim || c2<0 || c2>=out_dim) continue;
    if(r2==r1 && c2==c1) continue;
    float d_a = cosine_rows(net, L, r1, r2);
    float d_b = cosine_cols(net, L, c1, c2);
    float d_c = cosine_col_row(net, L, c1, r2);
    float d_d = cosine_row_col(net, L, r1, c2);
    float weight = powf(2.0f, sel.lambda*(fabsf(d_a) + fabsf(d_b) +
                                         fabsf(d_c) + fabsf(d_d)));
    float r = ((float)urand16() / 65535.0f) * w_max;
    if(r < weight){
      return off + c2*in_dim + r2;
    }
  }
  return idx_i;
}

// Perform up to FY_MAX_SWAPS locality-/hebbian-biased Fisher-Yates swaps inside a layer.
void fisherYatesMutate_Layer(BinaryNN &net, AxisSelector &sel, HCState &hc, uint8_t layer_idx){
  if (net.layer_count == 0) return;
  if (layer_idx >= net.layer_count) layer_idx = 0;

  uint16_t lo, hi;
  getLayerBitRange(net, layer_idx, lo, hi);
  if (hi <= lo) return;

  const uint16_t span  = (uint16_t)(hi - lo + 1);
  uint16_t start       = lo + (urand16() % span);

  for (uint8_t iter=0; iter<FY_MAX_SWAPS && (start + iter) < hi; ++iter){
    if (hc.swap_count >= SWAP_CAP) break;        // protect the log

    uint16_t i = start + iter;
    uint16_t j = selectSwapPartner(net, layer_idx, i, sel);
    if (i != j){
      net.swapWeights(i, j);
      hc.swap_from[hc.swap_count] = i;
      hc.swap_to[hc.swap_count]   = j;
      hc.swap_count++;
    }
  }
}


void revertSwaps(BinaryNN &net, HCState &hc){
  for(uint8_t k=0;k<hc.swap_count;++k){
    net.swapWeights(hc.swap_from[k], hc.swap_to[k]);
  }
}

void updateSelector(AxisSelector &sel, HCState &hc, bool positive){
  (void)hc; // no axis-specific bookkeeping anymore
  if(positive){
    if(sel.lambda < LAMBDA_MAX) sel.lambda++;
  } else {
    if(sel.lambda > 0) sel.lambda--;
  }
}

void backgroundBitFlip(BinaryNN &net, const AxisSelector &sel){
  // basis points: 0..10000
  const uint16_t P_MIN = 50;    // 0.5%
  const uint16_t P_MAX = 200;   // 2.0%
  uint16_t p_bg = P_MIN + ((uint16_t)(P_MAX - P_MIN) * (LAMBDA_MAX - sel.lambda)) / LAMBDA_MAX;

  uint16_t n = net.numWeightBits();
  for(uint16_t i=0;i<n;++i){
    if ((urand16() % 10000) < p_bg) {
      net.toggleWeight(i);
    }
  }
}

bool hillClimbStep(BinaryNN &net, HCState &hc, AxisSelector &sel, int16_t reward) {
  bool improved = false;
  if (hc.active) {
    hc.reward_accum += reward;
    hc.steps++;
    if (hc.steps >= HC_STEPS) {
      if (hc.reward_accum <= hc.reward_avg) {
        revertSwaps(net, hc);
        updateSelector(sel, hc, false);
      } else {
        updateSelector(sel, hc, true);
        improved = true;
      }
      if (hc.reward_avg == 0) {
        hc.reward_avg = hc.reward_accum;
      } else {
        hc.reward_avg += (hc.reward_accum - hc.reward_avg) >> HC_AVG_SHIFT;
      }
      hc.active = false;
    }
  }
  if (!hc.active) {
    hc.swap_count   = 0;   // start a fresh trial-wide log
    // Apply per-layer flips (kept inside each layer by fisherYatesMutate_Layer)
    for (uint8_t L = 0; L < net.layer_count; ++L) {
      fisherYatesMutate_Layer(net, sel, hc, L);
      if (hc.swap_count >= SWAP_CAP) break;  // safety
    }
    hc.active       = true;
    hc.reward_accum = 0;
    hc.steps        = 0;
  }
  return improved;
}

// -----------------------------
// Sensor and random helpers
// -----------------------------
int last_r=0, last_l=0, last_c=0;
bool last_button_pressed=false, last_plate_contact=false;

inline uint16_t urand16() {
  return ( (uint16_t)rand() << 1 ) ^ (uint16_t)rand();
}


// -----------------------------
// Feature extraction
// -----------------------------
uint8_t last_motor_bin = N_MOTOR/2;
uint8_t last_servo_bin = N_SERVO/2;
uint8_t last_beep_bin  = 0;

Feat read_features(const uint8_t *emb) {
  uint16_t l0 = analogRead(LDR_L_PIN);
  uint16_t r0 = analogRead(LDR_R_PIN);
  uint16_t c0 = analogRead(LDR_C_PIN);

  uint8_t l = (uint8_t)(l0 >> 2);
  uint8_t r = (uint8_t)(r0 >> 2);
  uint8_t c = (uint8_t)(c0 >> 2);
  bool plate_contact = (digitalRead(PLATE_PIN) == LOW);
  bool button_pressed = (digitalRead(BUTTON_PIN) == LOW);

  Feat F;
  F.x[0] = l;
  F.x[1] = r;
  F.x[2] = c;
  F.x[3] = plate_contact ? 255 : 0;
  F.x[4] = button_pressed ? 255 : 0;
  for (uint8_t i=0;i<EMB_DIM;++i) F.x[5+i] = emb[i];
  F.x[5+EMB_DIM] = 255; // bias
  for (uint8_t i = 5 + EMB_DIM + 1; i < FEAT_DIM; ++i) {
    F.x[i] = 0;  // padding
  }
  return F;
}

// -----------------------------
// Actuator helpers
// -----------------------------
uint8_t servo_bin_to_deg(uint8_t b){
  return (uint8_t)((uint16_t)b * 20);
}

inline void set_one_motor(int8_t dir, uint8_t inA, uint8_t inB, uint8_t en, uint8_t pwm){
  if (dir > 0){
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
    analogWrite(en, pwm);
  } else if (dir < 0){
    digitalWrite(inA, LOW);
    digitalWrite(inB, HIGH);
    analogWrite(en, pwm);
  } else {
    digitalWrite(inA, LOW);
    digitalWrite(inB, LOW);
    analogWrite(en, 0);
  }
}

inline void set_dual_motor(int8_t L_dir, int8_t R_dir, uint8_t pwm_on){
  set_one_motor(L_dir, IN1, IN2, ENA, pwm_on);
  set_one_motor(R_dir, IN3, IN4, ENB, pwm_on);
}

inline void set_motor_state9(uint8_t bin){
  if (bin >= 9) bin = 4;
  int8_t Ld = MOTOR9_DIRS[bin][0];
  int8_t Rd = MOTOR9_DIRS[bin][1];
  set_dual_motor(Ld, Rd, MOTOR_PWM_ON);
}

// -----------------------------
// Reward and PMI scaling
// -----------------------------
const int16_t DR_THRESH_Q8 = 16; // ~0.06 of full scale
inline uint8_t bin_dR_q8(int16_t dRq8){
  if (dRq8 > DR_THRESH_Q8) return 2;
  if (dRq8 < -DR_THRESH_Q8) return 0;
  return 1;
}

// Softmax model params
static int8_t beta_[3]            = {0,0,0};
static int8_t thetaM_[3][N_MOTOR] = {{0}};
static int8_t thetaS_[3][N_SERVO] = {{0}};
static int8_t thetaB_[3][N_BEEP]  = {{0}};
static int8_t gammaG_[3]          = {0,0,0};

const int8_t ETA_Q8 = 4;
const float LOGIT_SCALE = 32.0f;

inline float fexp_(float x){ return 1.0f + x + 0.5f*x*x; }

inline void predict_action_probs(uint8_t m, uint8_t s, uint8_t b, float P[3]){
  float z[3];
  for(int k=0;k<3;k++){
    float u = (beta_[k] + thetaM_[k][m] + thetaS_[k][s] + thetaB_[k][b]) / LOGIT_SCALE;
    z[k] = fexp_(u);
  }
  float Z = z[0]+z[1]+z[2];
  P[0]=z[0]/Z; P[1]=z[1]/Z; P[2]=z[2]/Z;
}

inline void predict_global_probs(float Pg[3]){
  float z0=fexp_(gammaG_[0]/LOGIT_SCALE);
  float z1=fexp_(gammaG_[1]/LOGIT_SCALE);
  float z2=fexp_(gammaG_[2]/LOGIT_SCALE);
  float Z=z0+z1+z2;
  Pg[0]=z0/Z; Pg[1]=z1/Z; Pg[2]=z2/Z;
}

const float PMI_CLAMP = 2.0f;
const float PMI_GAIN  = 1.0f;
inline float sigmoid_(float x){ return 1.0f/(1.0f+expf(-x)); }

float pmi_scale_and_update(uint8_t m, uint8_t s, uint8_t b, uint8_t kobs){
  float P[3], Pg[3]; predict_action_probs(m,s,b,P); predict_global_probs(Pg);
  float pmi = logf(P[kobs]) - logf(Pg[kobs]);
  if (pmi >  PMI_CLAMP) pmi =  PMI_CLAMP;
  if (pmi < -PMI_CLAMP) pmi = -PMI_CLAMP;
  float scale01 = sigmoid_(PMI_GAIN * pmi);
  for(int k=0;k<3;k++){
    float g = ((k==kobs)?1.0f:0.0f) - P[k];
    int8_t d = (int8_t)(g * ETA_Q8);
    beta_[k]       = constrain(beta_[k] + d, -96, 96);
    thetaM_[k][m]  = constrain(thetaM_[k][m] + d, -96, 96);
    thetaS_[k][s]  = constrain(thetaS_[k][s] + d, -96, 96);
    thetaB_[k][b]  = constrain(thetaB_[k][b] + d, -96, 96);
    gammaG_[k]     = constrain(gammaG_[k] + d, -96, 96);
  }
  return scale01;
}

int16_t compute_raw_reward_q8(){
  int r = analogRead(LDR_R_PIN);
  int l = analogRead(LDR_L_PIN);
  int c = analogRead(LDR_C_PIN);

  last_r = r; last_l = l; last_c = c;
  last_button_pressed = (digitalRead(BUTTON_PIN) == LOW);
  last_plate_contact = (digitalRead(PLATE_PIN) == LOW);

  int16_t rq8;
  if (last_plate_contact) {
    rq8 = PLATE_PENALTY_Q8;
  } else if (last_button_pressed) {
    rq8 = BUTTON_REWARD_Q8;
  } else {
    int total = l + r + c;
    rq8 = (int16_t)((int32_t)total * 256 / (3 * 1023));
    if (MOTOR9_DIRS[last_motor_bin][0] == 0 && MOTOR9_DIRS[last_motor_bin][1] == 0) {
      rq8 += MOTOR_IDLE_PENALTY_Q8;
    }
    if (rq8 > 256) rq8 = 256;
    if (rq8 < -256) rq8 = -256;
  }
  return rq8;
}

// -----------------------------
// Setup and loop
// -----------------------------
Servo servo;

static int16_t prev_raw_q8 = 0;

void setup(){
  pinMode(LDR_L_PIN, INPUT);
  pinMode(LDR_R_PIN, INPUT);
  pinMode(LDR_C_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(PLATE_PIN, INPUT_PULLUP);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(BEEP_PIN, OUTPUT);
  servo.attach(SERVO_PIN);
  servo.write(90);
  noTone(BEEP_PIN);
  Serial.begin(9600);

  // Embedding network and action network
  embed_net.addLayer(ACT_DIM, EMB_DIM);
  act_net.addLayer(FEAT_DIM, 24);
  act_net.addLayer(24, 24);
  act_net.addLayer(24, 24);
  act_net.addLayer(24, ACT_DIM);
  embed_net.initBuffers();
  act_net.initBuffers();
  Serial.print("SD init...");
  if (SD.begin(SD_CS_PIN)) {
    Serial.println("ok");
    sd_ready = true;
    if (!loadNetwork(embed_net, EMB_NET_FILE)) {
      Serial.println("Init embed_net");
      embed_net.randomizeWeights();
      saveNetwork(embed_net, EMB_NET_FILE);
    }
    if (!loadNetwork(act_net, ACT_NET_FILE)) {
      Serial.println("Init act_net");
      act_net.randomizeWeights();
      saveNetwork(act_net, ACT_NET_FILE);
    }
  } else {
    Serial.println("failed");
    embed_net.randomizeWeights();
    act_net.randomizeWeights();
  }
  initAxisSelector(sel_emb, embed_net.numWeightBits());
  initAxisSelector(sel_act, act_net.numWeightBits());
}

void loop(){
  // One-hot of previous action
  uint8_t one_hot[ACT_DIM];
  memset(one_hot, 0, sizeof(one_hot));
  one_hot[IDX_MOTOR0 + last_motor_bin] = 255;
  one_hot[IDX_SERVO0 + last_servo_bin] = 255;
  one_hot[IDX_BEEP0  + last_beep_bin]  = 255;

  // Run embedding network
  int16_t emb_scores[EMB_DIM];
  embed_net.forward(one_hot, emb_scores);
  uint8_t emb_feat[EMB_DIM];
  for (uint8_t i=0;i<EMB_DIM;++i) emb_feat[i] = emb_scores[i] >= 0 ? 255 : 0;

  // Build features and run action network
  Feat F = read_features(emb_feat);
  int16_t scores[ACT_DIM];
  act_net.forward(F.x, scores);

  // choose actions by argmax in each group
  int16_t best = -32768; uint8_t m_bin = 0;
  for (uint8_t i=0;i<N_MOTOR;++i){ if (scores[IDX_MOTOR0+i] > best){ best = scores[IDX_MOTOR0+i]; m_bin = i; } }
  best = -32768; uint8_t s_bin = 0;
  for (uint8_t i=0;i<N_SERVO;++i){ if (scores[IDX_SERVO0+i] > best){ best = scores[IDX_SERVO0+i]; s_bin = i; } }
  best = -32768; uint8_t b_bin = 0;
  for (uint8_t i=0;i<N_BEEP;++i){ if (scores[IDX_BEEP0+i] > best){ best = scores[IDX_BEEP0+i]; b_bin = i; } }

  set_motor_state9(m_bin);
  delay(100);
  servo.write(servo_bin_to_deg(s_bin));
  if (b_bin == 0) {
    noTone(BEEP_PIN);
  } else {
    for (uint8_t i=0;i<BEEP_TONES;++i){
      tone(BEEP_PIN, BEEP_FREQ[b_bin][i], BEEP_DUR[b_bin][i]);
      delay(BEEP_DUR[b_bin][i]);
    }
  }
  last_motor_bin = m_bin;
  last_servo_bin = s_bin;
  last_beep_bin  = b_bin;

  int16_t r_raw_q8 = compute_raw_reward_q8();
  int16_t dR_q8 = r_raw_q8 - prev_raw_q8;
  prev_raw_q8 = r_raw_q8;
  uint8_t kobs = bin_dR_q8(dR_q8);
  float scale01 = pmi_scale_and_update(m_bin, s_bin, b_bin, kobs);
  float scaled = (float)r_raw_q8 * scale01;
  if (scaled > 256.0f) scaled = 256.0f;
  if (scaled < -256.0f) scaled = -256.0f;
  int16_t r_q8 = (int16_t)scaled;

  bool emb_improved = hillClimbStep(embed_net, hc_emb, sel_emb, r_q8);
  bool act_improved = hillClimbStep(act_net, hc_act, sel_act, r_q8);
  backgroundBitFlip(embed_net,sel_emb);
  backgroundBitFlip(act_net,sel_act);
  if ((emb_improved || act_improved) && !sd_ready) {
    Serial.println("SD not ready");
  }
  if (sd_ready) {
    if (emb_improved) {
      saveNetwork(embed_net, EMB_NET_FILE);
    }
    if (act_improved) {
      saveNetwork(act_net, ACT_NET_FILE);
    }
  }

  Serial.print("r_q8="); Serial.println(r_q8);
  delay(20);
}

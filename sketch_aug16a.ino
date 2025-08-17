#include <Arduino.h>
#include <Servo.h>
#include <string.h>

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

// Fixed PWM speed for "on" states
const uint8_t MOTOR_PWM_ON = 255;

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
const uint8_t FEAT_DIM = 16;            // [ldrL, ldrR, ldrC, last action embedding (8), bias, padding]
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
  {100, 100, 100, 100, 200},
  {200, 100, 100, 100, 200},
  {150, 150, 300, 150, 150},
  {100, 200, 100, 200, 300},
  {300, 150, 150, 150, 300}
};

// -----------------------------
// Helper structs
// -----------------------------
struct Feat {
  uint8_t x[FEAT_DIM];
};

// -----------------------------
// Binary Feedforward Neural Network
// -----------------------------
class BinaryNN {
public:
  static const uint8_t MAX_LAYERS = 4;
  static const uint16_t MAX_WEIGHTS_BITS = 2560;
  static const uint8_t MAX_NODES = 32;

  struct Layer {
    uint8_t in_dim;
    uint8_t out_dim;
    uint16_t w_off;  // bit offset for weights
  };

  Layer layers[MAX_LAYERS];
  uint8_t layer_count = 0;
  uint16_t total_bits = 0;
  uint8_t weight_bytes[(MAX_WEIGHTS_BITS+7)/8];

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

  uint16_t numWeightBits() const { return total_bits; }

  void setWeightsFromBits(const uint8_t *bits) {
    uint16_t nB = (total_bits + 7) >> 3;
    memcpy(weight_bytes, bits, nB);
  }

  inline int8_t getWeight(uint16_t bit_index) const {
    return (weight_bytes[bit_index>>3] >> (bit_index & 7)) & 1 ? 1 : -1;
  }

  void forward(const uint8_t *input, int16_t *out) {
    int16_t bufA[MAX_NODES];
    int16_t bufB[MAX_NODES];
    for (uint8_t i=0;i<layers[0].in_dim;++i) bufA[i] = input[i];
    int16_t *cur_in = bufA;
    int16_t *cur_out = bufB;
    for (uint8_t l=0;l<layer_count;++l) {
      Layer &L = layers[l];
      uint8_t in_dim = L.in_dim;
      uint8_t out_dim = L.out_dim;
      uint16_t off = L.w_off;
      int16_t *next_out = (l==layer_count-1) ? out : cur_out;
      for (uint8_t j=0;j<out_dim;++j) {
        int16_t sum = 0;
        for (uint8_t i=0;i<in_dim;++i) {
          int8_t w = getWeight(off + j*in_dim + i);
          sum += w * cur_in[i];
        }
        next_out[j] = (l==layer_count-1) ? sum : (sum >= 0 ? 1 : -1);
      }
      if (l != layer_count-1) {
        int16_t *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
      }
    }
  }
};

BinaryNN act_net, embed_net;

// CGA probability models and buffers for both networks
uint8_t prob_act[BinaryNN::MAX_WEIGHTS_BITS];
uint8_t cur_bits_act[(BinaryNN::MAX_WEIGHTS_BITS+7)/8];
uint8_t prev_bits_act[(BinaryNN::MAX_WEIGHTS_BITS+7)/8];
uint8_t prob_emb[BinaryNN::MAX_WEIGHTS_BITS];
uint8_t cur_bits_emb[(BinaryNN::MAX_WEIGHTS_BITS+7)/8];
uint8_t prev_bits_emb[(BinaryNN::MAX_WEIGHTS_BITS+7)/8];
// Running average reward baseline
int16_t reward_avg_q8 = 0;
const uint8_t RAVG_BETA = 4;
bool have_prev = false;

// -----------------------------
// Sensor smoothing and random helpers
// -----------------------------
const uint8_t LDR_SMOOTH_SHIFT = 3; // 1/8 smoothing
uint16_t ldr_l_avg = 0, ldr_r_avg = 0, ldr_c_avg = 0;
int last_r=0, last_l=0, last_c=0;
bool last_button_pressed=false, last_plate_contact=false;

inline uint16_t smooth_analog_read(uint8_t pin, uint16_t &avg) {
  uint16_t raw = analogRead(pin);
  if (avg == 0) {
    avg = raw;
  } else {
    avg += ((int16_t)raw - (int16_t)avg) >> LDR_SMOOTH_SHIFT;
  }
  return avg;
}

inline uint16_t urand16() {
  return ( (uint16_t)rand() << 1 ) ^ (uint16_t)rand();
}

inline void sample_bits(uint8_t *bits, const uint8_t *prob, uint16_t n) {
  uint16_t nB = (n + 7) >> 3;
  memset(bits, 0, nB);
  for (uint16_t i=0;i<n;++i) {
    if ((uint8_t)(urand16() & 0xFF) < prob[i]) {
      bits[i>>3] |= (uint8_t)1 << (i & 7);
    }
  }
}

inline void cga_update(uint8_t *prob, const uint8_t *winner, const uint8_t *loser, uint16_t n) {
  for (uint16_t i=0;i<n;++i) {
    uint8_t w = (winner[i>>3] >> (i & 7)) & 1;
    uint8_t l = (loser[i>>3] >> (i & 7)) & 1;
    if (w != l) {
      if (w) { if (prob[i] < 255) prob[i]++; }
      else   { if (prob[i] >   0) prob[i]--; }
    }
  }
}

// -----------------------------
// Feature extraction
// -----------------------------
uint8_t last_motor_bin = N_MOTOR/2;
uint8_t last_servo_bin = N_SERVO/2;
uint8_t last_beep_bin  = 0;

Feat read_features(const uint8_t *emb) {
  uint16_t l0 = smooth_analog_read(LDR_L_PIN, ldr_l_avg);
  uint16_t r0 = smooth_analog_read(LDR_R_PIN, ldr_r_avg);
  uint16_t c0 = smooth_analog_read(LDR_C_PIN, ldr_c_avg);

  uint8_t l = (uint8_t)(l0 >> 2);
  uint8_t r = (uint8_t)(r0 >> 2);
  uint8_t c = (uint8_t)(c0 >> 2);

  Feat F;
  F.x[0] = l;
  F.x[1] = r;
  F.x[2] = c;
  for (uint8_t i=0;i<EMB_DIM;++i) F.x[3+i] = emb[i];
  F.x[3+EMB_DIM] = 255; // bias
  for (uint8_t i = 3 + EMB_DIM + 1; i < FEAT_DIM; ++i) {
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
// Reward
// -----------------------------
int16_t compute_reward_q8(){
  int r = smooth_analog_read(LDR_R_PIN, ldr_r_avg);
  int l = smooth_analog_read(LDR_L_PIN, ldr_l_avg);
  int c = smooth_analog_read(LDR_C_PIN, ldr_c_avg);

  last_r = r; last_l = l; last_c = c;
  last_button_pressed = (digitalRead(BUTTON_PIN) == LOW);
  last_plate_contact = (digitalRead(PLATE_PIN) == LOW);

  if (last_plate_contact) return PLATE_PENALTY_Q8;
  if (last_button_pressed) return BUTTON_REWARD_Q8;

  // Reward relative to ambient light (A3)
  int side_avg = (r + l) / 2;
  int diff = side_avg - c;
  int16_t r_q8 = (int16_t)((int32_t)diff * 256 / 1023);

  if (MOTOR9_DIRS[last_motor_bin][0] == 0 && MOTOR9_DIRS[last_motor_bin][1] == 0) {
    r_q8 += MOTOR_IDLE_PENALTY_Q8;
  }

  if (r_q8 > 256) r_q8 = 256;
  if (r_q8 < -256) r_q8 = -256;
  return r_q8;
}

// -----------------------------
// Setup and loop
// -----------------------------
Servo servo;

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
  for (uint16_t i=0;i<embed_net.numWeightBits();++i) prob_emb[i] = 128;
  for (uint16_t i=0;i<act_net.numWeightBits();++i)   prob_act[i] = 128;
}

void loop(){
  // One-hot of previous action
  uint8_t one_hot[ACT_DIM];
  memset(one_hot, 0, sizeof(one_hot));
  one_hot[IDX_MOTOR0 + last_motor_bin] = 255;
  one_hot[IDX_SERVO0 + last_servo_bin] = 255;
  one_hot[IDX_BEEP0  + last_beep_bin]  = 255;

  // Sample embedding network
  uint16_t nW_emb = embed_net.numWeightBits();
  sample_bits(cur_bits_emb, prob_emb, nW_emb);
  embed_net.setWeightsFromBits(cur_bits_emb);
  int16_t emb_scores[EMB_DIM];
  embed_net.forward(one_hot, emb_scores);
  uint8_t emb_feat[EMB_DIM];
  for (uint8_t i=0;i<EMB_DIM;++i) emb_feat[i] = emb_scores[i] >= 0 ? 255 : 0;

  // Build features and run action network
  Feat F = read_features(emb_feat);
  uint16_t nW_act = act_net.numWeightBits();
  sample_bits(cur_bits_act, prob_act, nW_act);
  act_net.setWeightsFromBits(cur_bits_act);
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
  delay(500);
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

  int16_t r_q8 = compute_reward_q8();
  int16_t adv_q8 = r_q8 - reward_avg_q8;
  reward_avg_q8 += (adv_q8 >> RAVG_BETA);

  if (have_prev){
    if (adv_q8 >= 0) {
      cga_update(prob_act, cur_bits_act, prev_bits_act, nW_act);
      cga_update(prob_emb, cur_bits_emb, prev_bits_emb, nW_emb);
    } else {
      cga_update(prob_act, prev_bits_act, cur_bits_act, nW_act);
      cga_update(prob_emb, prev_bits_emb, cur_bits_emb, nW_emb);
    }
  }
  uint16_t nB_act = (nW_act + 7) >> 3;
  uint16_t nB_emb = (nW_emb + 7) >> 3;
  memcpy(prev_bits_act, cur_bits_act, nB_act);
  memcpy(prev_bits_emb, cur_bits_emb, nB_emb);
  have_prev = true;

  Serial.print("r_q8="); Serial.println(r_q8);
  delay(20);
}


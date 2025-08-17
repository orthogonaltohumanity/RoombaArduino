#include <Arduino.h>
#include <Servo.h>

// -----------------------------
// Hardware pins (same as aug14b)
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

// Fixed PWM speed for "on" states (tweak as needed 0..255)
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
const uint8_t FEAT_DIM = 6;  // [ldrL, ldrR, ldrC, lastMotor, lastServo, bias]
const uint8_t ACT_DIM = N_MOTOR + N_SERVO + N_BEEP; // 24
const uint8_t IDX_MOTOR0 = 0;
const uint8_t IDX_SERVO0 = IDX_MOTOR0 + N_MOTOR;   // 9
const uint8_t IDX_BEEP0  = IDX_SERVO0 + N_SERVO;   // 19

// -----------------------------
// Reward parameters (same as aug14b)
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
// Ternary Recurrent Neural Network class
// -----------------------------
class TernaryRNN {
public:
  static const uint8_t MAX_LAYERS = 4;
  static const uint16_t MAX_WEIGHTS = 1024;
  static const uint8_t MAX_NODES = 32;
  struct Layer {
    uint8_t in_dim;
    uint8_t out_dim;
    bool recurrent;
    int8_t *w_in;
    int8_t *w_rec;
    int8_t state[MAX_NODES];
  };
  Layer layers[MAX_LAYERS];
  uint8_t layer_count = 0;
  int8_t weights[MAX_WEIGHTS];
  uint16_t offsets_in[MAX_LAYERS];
  uint16_t offsets_rec[MAX_LAYERS];
  uint16_t total_weights = 0;

  void addLayer(uint8_t in_dim, uint8_t out_dim, bool recurrent) {
    if (layer_count >= MAX_LAYERS) return;
    Layer &L = layers[layer_count];
    L.in_dim = in_dim;
    L.out_dim = out_dim;
    L.recurrent = recurrent;
    L.w_in = weights + total_weights;
    offsets_in[layer_count] = total_weights;
    total_weights += (uint16_t)in_dim * out_dim;
    if (recurrent) {
      L.w_rec = weights + total_weights;
      offsets_rec[layer_count] = total_weights;
      total_weights += (uint16_t)out_dim * out_dim;
      for (uint8_t i=0;i<out_dim;++i) L.state[i] = 0;
    } else {
      L.w_rec = nullptr;
      offsets_rec[layer_count] = total_weights;
    }
    layer_count++;
  }

  void resetState() {
    for (uint8_t l=0;l<layer_count;++l) {
      if (layers[l].recurrent) {
        for (uint8_t i=0;i<layers[l].out_dim;++i) layers[l].state[i] = 0;
      }
    }
  }

  uint16_t numWeights() const { return total_weights; }

  void setWeightsFromTrits(const int8_t *trits) {
    for (uint16_t i=0;i<total_weights;++i) {
      int8_t t = trits[i];
      weights[i] = (t > 0) ? 1 : (t < 0 ? -1 : 0);
    }
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
      uint16_t in_off = offsets_in[l];
      uint16_t rec_off = offsets_rec[l];
      int16_t *next_out = (l==layer_count-1) ? out : cur_out;
      for (uint8_t j=0;j<out_dim;++j) {
        int16_t sum = 0;
        for (uint8_t i=0;i<in_dim;++i) {
          int8_t w = weights[in_off + j*in_dim + i];
          sum += w * cur_in[i];
        }
        if (L.recurrent) {
          for (uint8_t i=0;i<out_dim;++i) {
            int8_t w = weights[rec_off + j*out_dim + i];
            sum += w * L.state[i];
          }
        }
        if (l==layer_count-1) {
          next_out[j] = sum;
        } else {
          if (sum > 0) next_out[j] = 1;
          else if (sum < 0) next_out[j] = -1;
          else next_out[j] = 0;
          if (L.recurrent) L.state[j] = (int8_t)next_out[j];
        }
      }
      if (l != layer_count-1) {
        int16_t *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
      }
    }
  }
};
TernaryRNN net;

// CGA probability model and buffers
uint8_t prob_neg[TernaryRNN::MAX_WEIGHTS]; // P(weight=-1)
uint8_t prob_pos[TernaryRNN::MAX_WEIGHTS]; // P(weight=+1)
int8_t cur_trits[TernaryRNN::MAX_WEIGHTS];
int8_t prev_trits[TernaryRNN::MAX_WEIGHTS];
// Running reward baseline
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

inline void sample_trits(int8_t *trits, uint16_t n) {
  for (uint16_t i=0;i<n;++i) {
    uint8_t r = (uint8_t)(urand16() & 0xFF);
    if (r < prob_neg[i]) trits[i] = -1;
    else if (r < (uint16_t)prob_neg[i] + prob_pos[i]) trits[i] = 1;
    else trits[i] = 0;
  }
}

inline void cga_update_trits(int8_t *winner, int8_t *loser, uint16_t n) {
  for (uint16_t i=0;i<n;++i) {
    int8_t w = winner[i], l = loser[i];
    if (w == l) continue;
    if (w == -1) {
      if (prob_neg[i] < 255) prob_neg[i]++;
      if (l == 1 && prob_pos[i] > 0) prob_pos[i]--;
    } else if (w == 1) {
      if (prob_pos[i] < 255) prob_pos[i]++;
      if (l == -1 && prob_neg[i] > 0) prob_neg[i]--;
    } else {
      if (l == -1 && prob_neg[i] > 0) prob_neg[i]--;
      else if (l == 1 && prob_pos[i] > 0) prob_pos[i]--;
    }
  }
}

// -----------------------------
// Feature extraction
// -----------------------------
uint8_t last_motor_bin = N_MOTOR/2;
uint8_t last_servo_bin = N_SERVO/2;
uint8_t last_beep_bin  = 0;

Feat read_features() {
  uint16_t l0 = smooth_analog_read(LDR_L_PIN, ldr_l_avg);
  uint16_t r0 = smooth_analog_read(LDR_R_PIN, ldr_r_avg);
  uint16_t c0 = smooth_analog_read(LDR_C_PIN, ldr_c_avg);

  uint8_t l = (uint8_t)(l0 >> 2);
  uint8_t r = (uint8_t)(r0 >> 2);
  uint8_t c = (uint8_t)(c0 >> 2);

  uint8_t lm = (uint8_t)( (uint16_t)last_motor_bin * 28 );
  uint8_t ls = (uint8_t)( (uint16_t)last_servo_bin * 25 );

  Feat F;
  F.x[0] = l;
  F.x[1] = r;
  F.x[2] = c;
  F.x[3] = lm;
  F.x[4] = ls;
  F.x[5] = 255; // bias
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

  // Reward relative to ambient light
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

  net.addLayer(FEAT_DIM, 16, true);
  net.addLayer(16, ACT_DIM, false);
  for (uint16_t i=0;i<net.numWeights();++i){ prob_neg[i] = 85; prob_pos[i] = 85; } // init ~1/3
}

void loop(){
  Feat F = read_features();
  uint16_t nW = net.numWeights();
  sample_trits(cur_trits, nW);
  net.setWeightsFromTrits(cur_trits);
  net.resetState();
  int16_t scores[ACT_DIM];
  net.forward(F.x, scores);

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
    if (adv_q8 >= 0) cga_update_trits(cur_trits, prev_trits, nW);
    else cga_update_trits(prev_trits, cur_trits, nW);
  }
  memcpy(prev_trits, cur_trits, nW);
  have_prev = true;

  Serial.print("r_q8="); Serial.println(r_q8);
  delay(20);
}


/*  Markov → Bandit → Φ-biased action selection with on-device learning
  Hardware: Arduino Uno R3

  Flow each loop:
    1) Read sensors & build 6-D feature vector x (uint8 scaled).
    2) Sample next_state ~ Markov(s_prev → ·) using integer counts.
    3) For that state, build action scores = bandit_score + Φ(x) per actuator.
    4) ε-greedy select: motor bin (9), servo bin (10), beep bin (5).
    5) Execute actuators.
    6) Compute reward r from LDRs; advantage = r - running_avg.
    7) Update:
       - Markov transition counts(s_prev, next_state)
       - Bandit counts(state, chosen bins)
       - Φ weights (perceptron-like, tiny step, clamped)
    8) Decay counts occasionally to avoid overflow.

  Pins:
    LDR_L_PIN A4, LDR_R_PIN A5, LDR_C_PIN A3, BUTTON_PIN 8, PLATE_PIN 9,
    SERVO_PIN A0, BEEP_PIN A1,
    ENA 6, IN1 2, IN2 3, ENB 5, IN3 4, IN4 7
*/

#include <Arduino.h>
#include <Servo.h>

// -----------------------------
// Hardware pins
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
// Order chosen to keep "stop" in the middle for intuition.
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
// Problem sizes (keep tiny)
// -----------------------------
const uint8_t N_STATE = 2;
const uint8_t N_MOTOR = 9;   // 9 speed bins (includes reverse/stop/forward)
const uint8_t N_SERVO = 10;  // 10 positions [0..180]
const uint8_t N_BEEP  = 5;   // 5 pitch bins
const uint8_t FEAT_DIM = 6;  // [ldrL, ldrR, ldrC, lastMotor, lastServo, bias]

struct Feat {
  uint8_t x[FEAT_DIM]; // scaled to [0..255]
};
struct Scores {
  // separate arrays for actuators
  int16_t motor[N_MOTOR];
  int16_t servo[N_SERVO];
  int16_t beep [N_BEEP];
};

// Action index layout for Φ (concatenate all actuator bins):
// [ motor 0..8 | servo 0..9 | beep 0..4 ]  => total 24
const uint8_t ACT_DIM = N_MOTOR + N_SERVO + N_BEEP; // 24
const uint8_t IDX_MOTOR0 = 0;
const uint8_t IDX_SERVO0 = IDX_MOTOR0 + N_MOTOR;           // 9
const uint8_t IDX_BEEP0  = IDX_SERVO0 + N_SERVO;           // 19

// -----------------------------
// Learning hyperparams
// -----------------------------
const uint8_t EPS_EGREEDY_PCT = 5;  // 5% explore
const uint8_t LR_PHI_NUM = 1;       // very small int step (scaled by feature)
const uint8_t DECAY_PERIOD = 50;    // every 50 steps, decay counts
const uint8_t DECAY_DIV = 2;        // halve counts (>=1)
const uint8_t BANDIT_REWARD_STEP = 2; // how many counts to add/sub (>=1)
const uint8_t MARKOV_REWARD_STEP = 1; // counts to add/sub (>=1)
const uint8_t REWARD_WINDOW = 4;     // apply accumulated rewards every 4 steps

const int16_t BUTTON_REWARD_Q8  = 256;  // reward when button is pressed

const int16_t PLATE_PENALTY_Q8 = -256; // penalty when collision plates touch

// -----------------------------
// State & learning structures
// -----------------------------

// Markov transition counts: C[s][s']  (uint8, kept >=1)
uint8_t T_count[N_STATE][N_STATE];

// Bandit counts per state for each actuator
uint8_t B_motor[N_STATE][N_MOTOR];
uint8_t B_servo[N_STATE][N_SERVO];
uint8_t B_beep [N_STATE][N_BEEP];

// Linear Φ weights: W[feat][act] as int8 (small, centered)
int8_t W_phi[FEAT_DIM][ACT_DIM];

// Running avg reward for advantage
int16_t rAvg_q8 = 0;   // fixed-point Q8 (r in [-256..+256] ~ [-1..+1])
const uint8_t RAVG_BETA = 4; // EMA: new = old + (r - old)/2^RAVG_BETA

// Bookkeeping
uint8_t cur_state = 0;
uint8_t prev_state = 0;

struct StepRecord {
  Feat F;
  Scores S;
  uint8_t prev_state;
  uint8_t cur_state;
  uint8_t m_bin;
  uint8_t s_bin;
  uint8_t b_bin;
};

StepRecord step_hist[REWARD_WINDOW];
uint8_t hist_idx = 0;
uint8_t steps_pending = 0;
int32_t adv_accum_q8 = 0;

uint8_t last_motor_bin = N_MOTOR/2; // start near stop
uint8_t last_servo_bin = N_SERVO/2; // mid
uint8_t last_beep_bin  = 0;

int last_r = 0;
int last_l = 0;
int last_c = 0;
bool last_button_pressed = false;
bool last_plate_contact = false;

Servo servo;

inline uint16_t urand16() {
  // Combine two rand() for more bits (rand() on AVR is 15-bit)
  return ( (uint16_t)rand() << 1 ) ^ (uint16_t)rand();
}

inline uint8_t randPct() {
  return (uint8_t)(urand16() % 100);
}

inline uint8_t clamp_u8(int16_t v, uint8_t lo, uint8_t hi){
  if (v < lo) return lo;
  if (v > hi) return hi;
  return (uint8_t)v;
}
inline int8_t clamp_i8(int16_t v, int8_t lo, int8_t hi){
  if (v < lo) return lo;
  if (v > hi) return hi;
  return (int8_t)v;
}

// -----------------------------
// Initialization
// -----------------------------
void init_counts() {
  // Initialize Markov counts with weakly uniform priors (>=1)
  for (uint8_t s=0; s<N_STATE; ++s){
    for (uint8_t sp=0; sp<N_STATE; ++sp){
      T_count[s][sp] = 2; // small bias
    }
  }
  // Bandits: small uniform priors
  for (uint8_t s=0;s<N_STATE;++s){
    for (uint8_t i=0;i<N_MOTOR;++i) B_motor[s][i] = 2;
    for (uint8_t i=0;i<N_SERVO;++i) B_servo[s][i] = 2;
    for (uint8_t i=0;i<N_BEEP ;++i) B_beep [s][i] = 2;
  }
  // Φ weights: small zeros around 0
  for (uint8_t f=0; f<FEAT_DIM; ++f){
    for (uint8_t a=0; a<ACT_DIM; ++a){
      W_phi[f][a] = 0;
    }
  }
}

// -----------------------------
// Sensors and features
// -----------------------------


Feat read_features() {
  int l0 = analogRead(LDR_L_PIN); // 0..1023
  int r0 = analogRead(LDR_R_PIN); // 0..1023
  int c0 = analogRead(LDR_C_PIN); // 0..1023

  // compress to 0..255
  uint8_t l = (uint8_t)(l0 >> 2);
  uint8_t r = (uint8_t)(r0 >> 2);
  uint8_t c = (uint8_t)(c0 >> 2);

  // last motor/servo as scaled bins (0..255)
  uint8_t lm = (uint8_t)( (uint16_t)last_motor_bin * 28 ); // 9*28=252
  uint8_t ls = (uint8_t)( (uint16_t)last_servo_bin * 25 ); // 10*25=250

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
// Markov sampling from integer counts
// -----------------------------
uint8_t sample_next_state(uint8_t s){
  // sample categorical proportional to T_count[s][·]
  uint16_t sum = 0;
  for (uint8_t sp=0; sp<N_STATE; ++sp) sum += T_count[s][sp];
  if (sum == 0) return s; // shouldn't happen

  uint16_t r = urand16() % sum;
  uint16_t acc = 0;
  for (uint8_t sp=0; sp<N_STATE; ++sp){
    acc += T_count[s][sp];
    if (r < acc) return sp;
  }
  return (uint8_t)(sum % N_STATE); // fallback
}

// -----------------------------
// Build action scores = bandit preference + Φ(x)
// We'll apply ε-greedy with a softmax sample over these logits.
// -----------------------------


void build_scores(uint8_t state, const Feat& F, Scores& S){
  // Bandit "preferences" = scaled counts / row_sum (approx)
  // Compute row sums:
  uint16_t sm=0, ss=0, sb=0;
  for (uint8_t i=0;i<N_MOTOR;++i) sm += B_motor[state][i];
  for (uint8_t i=0;i<N_SERVO;++i) ss += B_servo[state][i];
  for (uint8_t i=0;i<N_BEEP ;++i) sb += B_beep [state][i];
  if(!sm) sm=1; if(!ss) ss=1; if(!sb) sb=1;

  // Φ(x): compute per action column (dot product W[:,a]·x)
  // Keep it cheap: int16 acc (max 6 * 127 * 255 ~ 194k)
  // Then downscale to a small range by >> 8
  for (uint8_t i=0;i<N_MOTOR;++i){
    int16_t phi = 0;
    uint8_t a = IDX_MOTOR0 + i;
    for (uint8_t f=0; f<FEAT_DIM; ++f){
      phi += (int16_t)W_phi[f][a] * (int16_t)F.x[f];
    }
    phi >>= 8; // scale down
    int16_t band = (int16_t)((uint32_t)B_motor[state][i] * 256u / sm); // ~0..256
    S.motor[i] = band + phi;
  }
  for (uint8_t i=0;i<N_SERVO;++i){
    int16_t phi = 0;
    uint8_t a = IDX_SERVO0 + i;
    for (uint8_t f=0; f<FEAT_DIM; ++f){
      phi += (int16_t)W_phi[f][a] * (int16_t)F.x[f];
    }
    phi >>= 8;
    int16_t band = (int16_t)((uint32_t)B_servo[state][i] * 256u / ss);
    S.servo[i] = band + phi;
  }
  for (uint8_t i=0;i<N_BEEP;++i){
    int16_t phi = 0;
    uint8_t a = IDX_BEEP0 + i;
    for (uint8_t f=0; f<FEAT_DIM; ++f){
      phi += (int16_t)W_phi[f][a] * (int16_t)F.x[f];
    }
    phi >>= 8;
    int16_t band = (int16_t)((uint32_t)B_beep[state][i] * 256u / sb);
    S.beep[i] = band + phi;
  }
}

template<uint8_t N>
uint8_t argmax_i16(const int16_t *arr){
  int16_t best = arr[0];
  uint8_t idx = 0;
  for (uint8_t i=1;i<N;++i){
    if (arr[i] > best){ best = arr[i]; idx = i; }
  }
  return idx;
}

template<uint8_t N>
uint8_t random_index(){
  return (uint8_t)(urand16() % N);
}
template<uint8_t N>
uint8_t sample_from_logits(const int16_t *arr){
  // Softmax sampling with integer weights: p[i] ∝ (arr[i])^2
  uint32_t w[N];
  uint32_t sum = 0;
  for (uint8_t i=0; i<N; ++i){
    int32_t v = arr[i];
    uint32_t wi = (uint32_t)(v * v); // square to ensure non-negative weight
    w[i] = wi;
    sum += wi;
  }

  if (sum == 0) return (uint8_t)(urand16() % N); // defensive

  uint32_t r = ((uint32_t)urand16() << 16 | urand16()) % sum;
  uint32_t acc = 0;
  for (uint8_t i=0; i<N; ++i){
    acc += w[i];
    if (r < acc) return i;
  }
  return (uint8_t)(urand16() % N); // fallback
}

uint8_t pick_with_eps_greedy_motor(const Scores& S){
  if (randPct() < EPS_EGREEDY_PCT) return random_index<N_MOTOR>();
  return sample_from_logits<N_MOTOR>(S.motor);
}

uint8_t pick_with_eps_greedy_servo(const Scores& S){
  if (randPct() < EPS_EGREEDY_PCT) return random_index<N_SERVO>();
  return sample_from_logits<N_SERVO>(S.servo);
}

uint8_t pick_with_eps_greedy_beep(const Scores& S){
  if (randPct() < EPS_EGREEDY_PCT) return random_index<N_BEEP>();
  return sample_from_logits<N_BEEP>(S.beep);
}

// -----------------------------
// Actuator helpers
// -----------------------------


uint8_t servo_bin_to_deg(uint8_t b){
  // 10 bins across 0..180
  return (uint8_t)( (uint16_t)b * 20 ); // 0,20,40,...,180
}

// Creative beep patterns: frequency and duration for each of 5 tones per bin
const uint8_t BEEP_TONES = 5;
const uint16_t BEEP_FREQ[N_BEEP][BEEP_TONES] = {
  {262, 330, 392, 523, 659}, // C-major ascent (unused when b_bin==0)
  {659, 523, 392, 330, 262}, // mirror descent
  {262, 392, 523, 392, 262}, // rise then fall
  {330, 262, 330, 392, 523}, // low hop to high
  {523, 440, 392, 440, 523}  // bell curve
};

const uint16_t BEEP_DUR[N_BEEP][BEEP_TONES] = {
  {100, 100, 100, 100, 200}, // quick run with hold
  {200, 100, 100, 100, 200}, // long bookends
  {150, 150, 300, 150, 150}, // lingering center
  {100, 200, 100, 200, 300}, // growing finale
  {300, 150, 150, 150, 300}  // long ends
};

// Low-level: command one side given dir in {-1,0,+1} and pwm (0..255)
inline void set_one_motor(int8_t dir, uint8_t inA, uint8_t inB, uint8_t en, uint8_t pwm){
  if (dir > 0){        // forward
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
    analogWrite(en, pwm);
  } else if (dir < 0){ // backward
    digitalWrite(inA, LOW);
    digitalWrite(inB, HIGH);
    analogWrite(en, pwm);
  } else {             // off / brake
    digitalWrite(inA, LOW);
    digitalWrite(inB, LOW);
    analogWrite(en, 0);
  }
}

// High-level: set both motors by (L_dir, R_dir)
inline void set_dual_motor(int8_t L_dir, int8_t R_dir, uint8_t pwm_on){
  set_one_motor(L_dir, IN1, IN2, ENA, pwm_on);
  set_one_motor(R_dir, IN3, IN4, ENB, pwm_on);
}

// Entry point for bandit bin 0..8
inline void set_motor_state9(uint8_t bin){
  if (bin >= 9) bin = 4; // safety: default to STOP
  int8_t Ld = MOTOR9_DIRS[bin][0];
  int8_t Rd = MOTOR9_DIRS[bin][1];
  set_dual_motor(Ld, Rd, MOTOR_PWM_ON);
}


// -----------------------------
// Reward
// -----------------------------
int16_t compute_reward_q8(){ // returns Q8 fixed in [-256..256]
  // Read sensors up front and remember values for reporting.
  int r = analogRead(LDR_R_PIN); // A5 0..1023
  int l = analogRead(LDR_L_PIN); // A4 0..1023
  int c = analogRead(LDR_C_PIN); // A3 0..1023

  last_r = r;
  last_l = l;
  last_c = c;
  last_button_pressed = (digitalRead(BUTTON_PIN) == LOW);
  last_plate_contact = (digitalRead(PLATE_PIN) == LOW);

  // If collision plates touch, return fixed penalty.
  if (last_plate_contact) {
    return PLATE_PENALTY_Q8;
  }

  // If reward button is pressed, return fixed reward.
  if (last_button_pressed) {
    return BUTTON_REWARD_Q8;

  }

  // Reward idea: balance between A4 & A5 plus brightness on A3.
  int clos = 1023 - abs(l - r);  // 0..1023 (higher when closer)

  int16_t clos_q8 = (int16_t)((uint32_t)clos * 256u / 1023u); // 0..256
  int16_t c_q8    = (int16_t)((uint32_t)c    * 256u / 1023u); // 0..256

  int16_t r_q8 = clos_q8 + c_q8 - 256; // [-256..+256]
  if (r_q8 > 256) r_q8 = 256;
  if (r_q8 < -256) r_q8 = -256;
  return r_q8;
}

// -----------------------------
// Updates
// -----------------------------

void update_markov(uint8_t s_prev, uint8_t s_cur, int16_t adv_q8){
  if (adv_q8 >= 0){
    uint16_t v = T_count[s_prev][s_cur] + MARKOV_REWARD_STEP;
    T_count[s_prev][s_cur] = (v > 255) ? 255 : (uint8_t)v;
  } else {
    // penalize slightly but keep >=1
    int16_t v = (int16_t)T_count[s_prev][s_cur] - (int16_t)MARKOV_REWARD_STEP;
    if (v < 1) v = 1;
    T_count[s_prev][s_cur] = (uint8_t)v;
  }
}

void update_bandits(uint8_t state, uint8_t bm, uint8_t bs, uint8_t bb, int16_t adv_q8){
  // reinforce chosen bins
  uint8_t step = BANDIT_REWARD_STEP;
  if (adv_q8 >= 0){
    uint16_t v;

    v = B_motor[state][bm] + step; B_motor[state][bm] = (v>255)?255:(uint8_t)v;
    v = B_servo[state][bs] + step; B_servo[state][bs] = (v>255)?255:(uint8_t)v;
    v = B_beep [state][bb] + step; B_beep [state][bb] = (v>255)?255:(uint8_t)v;
  } else {
    int16_t v;

    v = (int16_t)B_motor[state][bm] - (int16_t)step; if (v<1) v=1; B_motor[state][bm]=(uint8_t)v;
    v = (int16_t)B_servo[state][bs] - (int16_t)step; if (v<1) v=1; B_servo[state][bs]=(uint8_t)v;
    v = (int16_t)B_beep [state][bb] - (int16_t)step; if (v<1) v=1; B_beep [state][bb] =(uint8_t)v;
  }
}

// Perceptron-like Φ update: w[:,a*] += sign(adv)*lr * x  ; w[:,competing] -= small part
// Keep super tiny to avoid oscillations; clamp to [-127..127].
void update_phi(const Feat& F, uint8_t a_motor, uint8_t a_servo, uint8_t a_beep, int16_t adv_q8, const Scores& S){
  int8_t sg = (adv_q8 >= 0) ? 1 : -1;

  // For each actuator, also find strongest competing action to push away a bit.
  auto top_other = [](const int16_t* arr, uint8_t N, uint8_t chosen)->uint8_t{
    int16_t best = INT16_MIN; uint8_t idx=0;
    for (uint8_t i=0;i<N;++i){
      if (i==chosen) continue;
      if (arr[i] > best){ best = arr[i]; idx = i; }
    }
    return idx;
  };

  // Motor segment
  uint8_t comp_m = top_other(S.motor, N_MOTOR, a_motor);
  uint8_t aM = IDX_MOTOR0 + a_motor;
  uint8_t cM = IDX_MOTOR0 + comp_m;

  // Servo segment
  uint8_t comp_s = top_other(S.servo, N_SERVO, a_servo);
  uint8_t aS = IDX_SERVO0 + a_servo;
  uint8_t cS = IDX_SERVO0 + comp_s;

  // Beep segment
  uint8_t comp_b = top_other(S.beep, N_BEEP, a_beep);
  uint8_t aB = IDX_BEEP0 + a_beep;
  uint8_t cB = IDX_BEEP0 + comp_b;

  // Apply tiny updates
  for (uint8_t f=0; f<FEAT_DIM; ++f){
    int16_t dw = (int16_t)sg * (int16_t)LR_PHI_NUM * (int16_t)(F.x[f] >> 5); // scale x by 1/32
    // chosen gets +dw
    W_phi[f][aM] = clamp_i8( (int16_t)W_phi[f][aM] + dw, -127, 127);
    W_phi[f][aS] = clamp_i8( (int16_t)W_phi[f][aS] + dw, -127, 127);
    W_phi[f][aB] = clamp_i8( (int16_t)W_phi[f][aB] + dw, -127, 127);

    // competing gets -dw/2 (gentle)
    int16_t dwc = (dw >> 1);
    W_phi[f][cM] = clamp_i8( (int16_t)W_phi[f][cM] - dwc, -127, 127);
    W_phi[f][cS] = clamp_i8( (int16_t)W_phi[f][cS] - dwc, -127, 127);
    W_phi[f][cB] = clamp_i8( (int16_t)W_phi[f][cB] - dwc, -127, 127);
  }
}

// -----------------------------
// Decay (prevent overflow, maintain plasticity)
// -----------------------------
void decay_counts(){
  // Markov
  for (uint8_t s=0;s<N_STATE;++s){
    for (uint8_t sp=0;sp<N_STATE;++sp){
      uint8_t v = T_count[s][sp];
      v = (uint8_t)max(1, (int)v / DECAY_DIV);
      T_count[s][sp] = v;
    }
  }
  // Bandits
  for (uint8_t s=0;s<N_STATE;++s){
    for (uint8_t i=0;i<N_MOTOR;++i){
      uint8_t v = B_motor[s][i];
      v = (uint8_t)max(1, (int)v / DECAY_DIV);
      B_motor[s][i] = v;
    }
    for (uint8_t i=0;i<N_SERVO;++i){
      uint8_t v = B_servo[s][i];
      v = (uint8_t)max(1, (int)v / DECAY_DIV);
      B_servo[s][i] = v;
    }
    for (uint8_t i=0;i<N_BEEP;++i){
      uint8_t v = B_beep[s][i];
      v = (uint8_t)max(1, (int)v / DECAY_DIV);
      B_beep[s][i] = v;
    }
  }
}

// -----------------------------
// Setup & Loop
// -----------------------------
void setup(){
  Serial.begin(115200);
  randomSeed(analogRead(A7) ^ micros()); // a little entropy: A7 floating on Uno is fine

  pinMode(LDR_L_PIN, INPUT);
  pinMode(LDR_R_PIN, INPUT);
  pinMode(LDR_C_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(PLATE_PIN, INPUT_PULLUP);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(BEEP_PIN, OUTPUT);

  servo.attach(SERVO_PIN);

  init_counts();

  // Start centered/quiet
  servo.write(90);
  noTone(BEEP_PIN);

  Serial.println(F("[MB-Φ] Ready."));
  
}

void loop(){
  static uint32_t step = 0;

  // 1) Features
  Feat F = read_features();

  // 2) Markov step: sample next internal state
  prev_state = cur_state;
  cur_state = sample_next_state(prev_state);

  // 3) Build scores (bandit + Φ) for the chosen state
  Scores S;
  build_scores(cur_state, F, S);

  // 4) ε-greedy selections
  uint8_t m_bin = pick_with_eps_greedy_motor(S);
  uint8_t s_bin = pick_with_eps_greedy_servo(S);
  uint8_t b_bin = pick_with_eps_greedy_beep (S);

  // 5) Execute actions
  set_motor_state9(m_bin);
  delay(500);
  uint8_t sv_deg = servo_bin_to_deg(s_bin);
  servo.write(sv_deg);

  if (b_bin == 0){
    noTone(BEEP_PIN);
  } else {
    for (uint8_t i = 0; i < BEEP_TONES; ++i){
      uint16_t f = BEEP_FREQ[b_bin][i];
      uint16_t d = BEEP_DUR[b_bin][i];
      tone(BEEP_PIN, f, d);
      delay(d);
    }
  }

  // remember for features
  last_motor_bin = m_bin;
  last_servo_bin = s_bin;
  last_beep_bin  = b_bin;

  delay(20); // allow sensors to respond a bit

  // store step for delayed reward
  StepRecord &rec = step_hist[hist_idx];
  rec.F = F;
  rec.S = S;
  rec.prev_state = prev_state;
  rec.cur_state = cur_state;
  rec.m_bin = m_bin;
  rec.s_bin = s_bin;
  rec.b_bin = b_bin;
  hist_idx = (hist_idx + 1) % REWARD_WINDOW;
  steps_pending++;

  // 6) Reward & advantage
  int16_t r_q8 = compute_reward_q8();         // ~[-256..+256]
  if(b_bin == 0 && r_q8 > 0){
    r_q8 += 100;
  }
  int16_t adv_q8 = r_q8 - rAvg_q8;
  // Update running average
  rAvg_q8 += (adv_q8 >> RAVG_BETA);
  adv_accum_q8 += adv_q8;

  if (steps_pending >= REWARD_WINDOW){
    int16_t batch_adv = (int16_t)max(min(adv_accum_q8, (int32_t)32767), (int32_t)-32768);
    for (uint8_t i=0;i<REWARD_WINDOW;++i){
      StepRecord &sr = step_hist[i];
      update_markov(sr.prev_state, sr.cur_state, batch_adv);
      update_bandits(sr.cur_state, sr.m_bin, sr.s_bin, sr.b_bin, batch_adv);
      update_phi(sr.F, sr.m_bin, sr.s_bin, sr.b_bin, batch_adv, sr.S);
    }
    adv_accum_q8 = 0;
    steps_pending = 0;
    hist_idx = 0;
  }

  // 8) Periodic decay
  if ((++step % DECAY_PERIOD) == 0){
    decay_counts();
  }

  // Debug (lightweight)
  if ((step % 25) == 0){
    Serial.print(F("s=")); Serial.print(cur_state);
    Serial.print(F(" m=")); Serial.print(m_bin);
    Serial.print(F(" sv=")); Serial.print(s_bin);
    Serial.print(F(" be=")); Serial.print(b_bin);
    Serial.print(F(" r=")); Serial.print(r_q8);
    Serial.print(F(" adv=")); Serial.println(adv_q8);
  }

  if (last_plate_contact) {
    Serial.print(F("Plate contact "));
  }
  if (last_button_pressed) {
    Serial.print(F("Button pressed "));
  }
  Serial.print(F("A5=")); Serial.print(last_r);
  Serial.print(F(" A4=")); Serial.print(last_l);
  Serial.print(F(" A3=")); Serial.println(last_c);
}

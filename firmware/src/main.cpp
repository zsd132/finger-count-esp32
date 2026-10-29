/*
 * ESP32 + 0.96" SSD1306 OLED (I2C 128x64)
 *
 * 电脑端通过 USB 串口发送（115200 8N1，一行一帧）：
 *     FC <总数> <手数> [<手指数>[:L|R] ...]
 * 例如：
 *     FC 0 0
 *     FC 3 1 3:R
 *     FC 5 2 2:L 3:R
 *     FC 4 1 4          （左右手未知，YOLO 引擎）
 *
 * 接线（ESP32 DevKit / WROOM-32）：
 *     OLED VCC -> 3V3
 *     OLED GND -> GND
 *     OLED SCL -> GPIO22
 *     OLED SDA -> GPIO21
 *
 * 依赖库：U8g2（PlatformIO 会自动装，见 platformio.ini）
 */

#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>

// ---- 如果你不是普通 ESP32（C3/S3），改这里 ---------------------------
static const int I2C_SDA = 21;
static const int I2C_SCL = 22;
// ---------------------------------------------------------------------

// ---- 输出引脚：手指数量 == 5 时输出高电平，其余（含 0）输出低电平 ----
static const int OUTPUT_PIN = 17;
// ---------------------------------------------------------------------

U8G2_SSD1306_128X64_NONAME_F_HW_I2C display(
    U8G2_R0, U8X8_PIN_NONE, I2C_SCL, I2C_SDA);

char lineBuf[96];
uint8_t lineLen = 0;

int fingerCount = -1;      // -1 = 还没收到电脑数据
int handCount = 0;
int handSides[2] = {0, 0}; // 0=未知, 1=L, 2=R
unsigned long lastRxMs = 0;
bool dirty = true;
bool lastOffline = true;

void drawBadge(int x, int side) {
  display.drawRFrame(x, 1, 13, 13, 3);
  display.setFont(u8g2_font_6x12_tf);
  const char *t = (side == 1) ? "L" : (side == 2) ? "R" : "?";
  display.drawStr(x + 4, 11, t);
}

void drawScreen(bool offline) {
  display.clearBuffer();

  // 左上角：在线/离线状态
  display.setFont(u8g2_font_6x12_tf);
  display.drawStr(2, 11, offline ? "NO LINK" : "FINGERS");

  // 右上角：左右手徽标
  int n = handCount;
  if (n > 2) n = 2;
  if (n == 1) {
    drawBadge(114, handSides[0]);
  } else if (n >= 2) {
    drawBadge(100, handSides[0]);
    drawBadge(114, handSides[1]);
  }

  // 中间大数字
  display.setFont(u8g2_font_logisoso46_tn);
  char buf[8];
  if (fingerCount < 0) {
    strcpy(buf, "--");
  } else {
    snprintf(buf, sizeof(buf), "%d", fingerCount);
  }
  int w = display.getStrWidth(buf);
  display.drawStr((128 - w) / 2, 58, buf);

  display.sendBuffer();
}

void handleLine(char *s) {
  if (strncmp(s, "FC ", 3) != 0) return;

  lastRxMs = millis();
  if (strcmp(s, "FC READY") == 0) {
    dirty = true;
    return;
  }

  char *p = s + 3;
  int total = 0, hands = 0, consumed = 0;
  if (sscanf(p, "%d %d%n", &total, &hands, &consumed) != 2) return;
  p += consumed;

  int sides[2] = {0, 0};
  int idx = 0;
  char *tok = strtok(p, " ");
  while (tok != NULL && idx < 2) {
    int c = 0;
    char side = 0;
    if (sscanf(tok, "%d:%c", &c, &side) == 2) {
      sides[idx] = (side == 'L' || side == 'l') ? 1
                 : (side == 'R' || side == 'r') ? 2
                 : 0;
    }
    idx++;
    tok = strtok(NULL, " ");
  }

  if (total < 0) total = 0;
  if (total > 10) total = 10;
  if (hands < 0) hands = 0;

  // 5 根手指 -> 高电平；0~4 根 -> 低电平
  digitalWrite(OUTPUT_PIN, (total == 5) ? HIGH : LOW);

  bool changed = (total != fingerCount) || (hands != handCount) ||
                 (sides[0] != handSides[0]) || (sides[1] != handSides[1]);
  if (changed) {
    fingerCount = total;
    handCount = hands;
    handSides[0] = sides[0];
    handSides[1] = sides[1];
    dirty = true;
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, LOW);   // 上电默认低电平

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000L);
  display.begin();
  display.setContrast(255);

  display.clearBuffer();
  display.setFont(u8g2_font_6x12_tf);
  display.drawStr(10, 24, "FINGER COUNTER");
  display.drawStr(18, 42, "waiting for PC");
  display.sendBuffer();

  lastRxMs = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      handleLine(lineBuf);
      lineLen = 0;
    } else if (c != '\r') {
      if (lineLen < sizeof(lineBuf) - 1) {
        lineBuf[lineLen++] = c;
      } else {
        lineLen = 0;
      }
    }
  }

  bool offline = (lastRxMs == 0) || (millis() - lastRxMs > 3000);
  if (offline != lastOffline) {
    lastOffline = offline;
    dirty = true;
  }
  if (dirty) {
    drawScreen(offline);
    dirty = false;
  }
}

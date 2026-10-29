// 手指计数接收端示例（Arduino / ESP32 通用）
//
// 电脑端发送格式（115200 8N1，一行一帧）：
//   FC <total> <n_hands> [count_1 count_2 ...]\n
// 例如：
//   FC 0 0          没有检测到手
//   FC 3 1 3        一只手，3 根手指
//   FC 5 2 2 3      两只手，分别 2 和 3 根
//
// 把 total 显示到数码管 / OLED / LED 即可；本示例先用串口回显。

char line[80];
uint8_t len = 0;

void setup() {
  Serial.begin(115200);
  // Leonardo / Micro / ESP32-S3 等原生 USB 串口可等主机连接；
  // Uno 等会因打开串口自动复位，电脑端已留 1.5 秒启动时间。
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      line[len] = '\0';
      handleFrame(line);
      len = 0;
    } else if (c != '\r') {
      if (len < sizeof(line) - 1) {
        line[len++] = c;
      } else {
        len = 0;  // 溢出丢弃，等待下一帧
      }
    }
  }
}

void handleFrame(char *s) {
  if (strncmp(s, "FC ", 3) != 0) return;

  if (strcmp(s, "FC READY") == 0) {
    Serial.println(F("MCU: link ready"));
    return;
  }

  int total = 0, hands = 0;
  if (sscanf(s + 3, "%d %d", &total, &hands) != 2) return;

  // ---- 在这里接你的显示/执行机构 ------------------------------------
  // 例：TM1637 数码管
  //   display.showNumberDec(total);
  // 例：OLED
  //   oled.clearDisplay(); oled.setCursor(0,0);
  //   oled.print(total); oled.display();
  // 例：点亮 total 个 LED
  for (int i = 0; i < 6; i++) {
    digitalWrite(2 + i, i < total ? HIGH : LOW);
  }

  Serial.print(F("total="));
  Serial.print(total);
  Serial.print(F(" hands="));
  Serial.println(hands);
}

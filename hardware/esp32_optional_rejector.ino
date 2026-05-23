// Optional future hardware sketch for the smart manufacturing prototype.
// This is not required for the AI model or dashboard to work.
//
// Idea:
// - Python predicts good/rotten.
// - If rotten, Python can send the text "REJECT" over serial.
// - ESP32 moves a servo to push the item aside.

#include <ESP32Servo.h>

Servo rejectServo;

const int SERVO_PIN = 18;
const int REST_ANGLE = 0;
const int REJECT_ANGLE = 70;
const int REJECT_DELAY_MS = 600;

void setup() {
  Serial.begin(115200);
  rejectServo.attach(SERVO_PIN);
  rejectServo.write(REST_ANGLE);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "REJECT") {
      rejectServo.write(REJECT_ANGLE);
      delay(REJECT_DELAY_MS);
      rejectServo.write(REST_ANGLE);
      Serial.println("DONE");
    }
  }
}

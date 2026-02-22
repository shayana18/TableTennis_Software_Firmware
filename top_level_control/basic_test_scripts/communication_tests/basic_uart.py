import re
import serial
import struct
import time

PORT = '/dev/tty.usbmodem144203'
BAUD_RATE = 115200
FLOAT_COUNT = 7
BYTES_PER_FLOAT = 4
PAYLOAD_LEN = FLOAT_COUNT * BYTES_PER_FLOAT


def read_response(port, max_wait_s=1.0, settle_s=0.2):
    data = bytearray()
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        chunk = port.read(128)
        if chunk:
            data.extend(chunk)
            deadline = time.time() + settle_s
    return bytes(data)


if __name__ == "__main__":

    serial_port = serial.Serial(port=PORT, baudrate=BAUD_RATE, timeout=1)

    print("Enter 7 floats separated by commas (e.g. 1,2,3,4,5,6,7).")
    print("You can also enter a single float to send it and fill the rest with 0.0.")

    while True:
        raw = input("Provide floats to send over UART: ").strip()
        if not raw:
            continue

        try:
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            values = [float(p) for p in parts]
        except ValueError:
            print("Invalid input. Use comma-separated floats.\n")
            continue

        if len(values) == 1:
            values = values + [0.0] * (FLOAT_COUNT - 1)
        elif len(values) != FLOAT_COUNT:
            print(f"Invalid input. Provide 1 or {FLOAT_COUNT} floats.\n")
            continue

        payload = struct.pack("<" + "f" * FLOAT_COUNT, *values)

        serial_port.reset_input_buffer()
        serial_port.write(payload)
        serial_port.flush()

        response = read_response(serial_port)
        if not response:
            print("no response from stm32\n")
            continue

        text = response.decode(errors="replace")
        print(f"sent:   {values}")

        def extract_int(pattern):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return None

        target_type_match = re.search(r"target type\s+([A-Z_]+)", text, re.IGNORECASE)
        target_type = target_type_match.group(1).upper() if target_type_match else "UNKNOWN"

        int_x = extract_int(r"intercept x:\s*(-?\d+)")
        int_y = extract_int(r"intercept y:\s*(-?\d+)")
        int_z = extract_int(r"intercept z:\s*(-?\d+)")
        int_t = extract_int(r"interception time:\s*(-?\d+)")
        int_sent = extract_int(r"time_sent\s*(-?\d+)")
        int_stamp = extract_int(r"time stamp\s*(-?\d+)")

        print(f"target_type: {target_type}")
        print("scaled values (1/1000):")
        print(f"  x: {int_x / 1000.0 if int_x is not None else 'NA'}")
        print(f"  y: {int_y / 1000.0 if int_y is not None else 'NA'}")
        print(f"  z: {int_z / 1000.0 if int_z is not None else 'NA'}")
        print(f"  intercept_time: {int_t / 1000.0 if int_t is not None else 'NA'}")
        print(f"  time_sent: {int_sent / 1000.0 if int_sent is not None else 'NA'}")
        print(f"  timestamp: {int_stamp / 1000.0 if int_stamp is not None else 'NA'}")
        print()

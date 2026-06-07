import serial
import pynmea2

# Open serial port (use /dev/serial0 or /dev/ttyS0 depending on Pi model)
port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

while True:
    try:
        line = port.readline().decode('ascii', errors='replace')
        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
            msg = pynmea2.parse(line)
            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                print(f"Latitude: {msg.latitude}, Longitude: {msg.longitude}")
    except pynmea2.ParseError:
        continue
    except KeyboardInterrupt:
        print("Exiting...")
        break

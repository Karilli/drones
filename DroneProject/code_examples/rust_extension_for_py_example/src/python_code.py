import ctypes

rust_lib = ctypes.CDLL("..\\target\\release\\rust_lib.dll")


if __name__ == "__main__":
    BYTES = "Python says hi inside Rust!".encode("utf-8")
    rust_lib.print_string(BYTES)

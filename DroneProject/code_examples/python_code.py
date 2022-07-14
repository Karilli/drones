import ctypes

rust_lib_path = "..\\target\\release\\rust_lib.dll"
rust_lib = ctypes.CDLL(rust_lib_path)


if __name__ == "__main__":
    SOME_BYTES = "Python says hi inside Rust!".encode("utf-8")
    rust_lib.print_string(SOME_BYTES)

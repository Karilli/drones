import ctypes

rust_lib_name = "..\\target\\release\\rust_lib.dll"
rust = ctypes.CDLL(rust_lib_name)


if __name__ == "__main__":
    SOME_BYTES = "Python says hi inside Rust!".encode("utf-8")
    rust.print_string(SOME_BYTES)

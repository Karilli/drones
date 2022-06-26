import ctypes

rust_lib_name = "rust_lib.dll"
rust = ctypes.CDLL("target/release/{rust_lib_name}")


if __name__ == "__main__":
    SOME_BYTES = "Python says hi inside Rust!".encode("utf-8")
    rust.print_string(SOME_BYTES)
